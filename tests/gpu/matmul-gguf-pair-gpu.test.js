import assert from 'node:assert/strict';

// Pair GEMV (matmul_gguf_pair_f32.wgsl) against the dequantized CPU reference: Q4_0,
// Q8_0 and Q6_K, M = 1..7, an offset weight bank, offset output with guarded prefix and tail,
// immediate and recorded execution. Odd block counts are refused.

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`matmul-gguf-pair-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
if (!getKernelCapabilities()?.hasSubgroups) {
  console.log('matmul-gguf-pair-gpu.test: skipped (no subgroups)');
  destroyDevice();
  process.exit(0);
}
const { acquireBuffer, releaseBuffer, uploadData, readBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { createCommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { runQ40Matmul, recordQ40Matmul } = await import('../../src/gpu/kernels/q4_0.js');
const { packedFixture, decodePacked, projectReference } = await import('../helpers/q4_0-fixture.js');

const device = getDevice();
const allocated = [];
const upload = (data) => {
  const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
  padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = acquireBuffer(padded.byteLength);
  uploadData(buffer, padded);
  allocated.push(buffer);
  return buffer;
};
const errors = [];
device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));

try {
  for (const [dtype, blockBytes, blockValues] of [['q4_0', 18, 32], ['q8_0', 34, 32], ['q6k', 210, 256]]) {
    const shapes = blockValues === 32
      ? [[1, 7, 64, 0], [3, 9, 256, 1], [4, 19, 512, 1], [5, 19, 512, 1], [7, 19, 2048, 0], [1, 130, 2048, 0], [2, 33, 4096, 1]]
      : [[1, 7, 512, 0], [3, 9, 512, 1], [4, 19, 512, 1], [5, 19, 2048, 1], [7, 19, 5120, 0], [1, 130, 2048, 0], [2, 33, 2048, 1]];
    for (const [M, N, K, bank] of shapes) {
      const bytes = packedFixture(dtype, (bank + 1) * N, K);
      const decoded = decodePacked(dtype, bytes, (bank + 1) * N, K);
      const weight = createWeightBuffer(upload(bytes), dtype, 'row', bank ? [2, N, K] : [N, K]);
      const values = Float32Array.from({ length: (M + bank) * K }, (_, i) => Math.sin(i * 0.23) * 0.125);
      const input = createTensor(upload(values), 'f32', [M + bank, K]);
      const expected = projectReference(values.subarray(bank * K), decoded.subarray(bank * N * K), M, N, K);
      for (const mode of ['immediate', 'recorded']) {
        let baselineBits = null;
        for (const variant of dtype === 'q4_0' ? ['q4_0_pair', 'q4_0_pair_rows2', 'q4_0_pair_rows4', 'q4_0_pair_columns', 'q4_0_pair_columns_rows4', 'q4_0_pair_columns_hoisted']
          : dtype === 'q8_0' ? ['q8_0_pair', 'q8_0_pair_columns', 'q8_0_pair_columns_rows4', 'q8_0_pair_columns_hoisted'] : ['q6k_pair', 'q6k_pair_columns', 'q6k_pair_columns_wg128_rows1', 'q6k_pair_columns_hoisted']) {
          const recorder = mode === 'recorded' ? createCommandRecorder('matmul_gguf_pair') : null;
          const prefix = bank * 4;
          const output = upload(new Float32Array(M * N + 16).fill(-12345));
          const options = { outputDtype: 'f32', variant, outputBuffer: output,
            aOffset: bank * K * 4, bOffset: bank * N * K / blockValues * blockBytes, cOffset: prefix * 4 };
          const result = recorder
            ? await recordQ40Matmul(recorder, input, weight, M, N, K, options)
            : await runQ40Matmul(input, weight, M, N, K, options);
          if (recorder) await recorder.submitAndWait();
          const actual = new Float32Array(await readBuffer(result.buffer, (M * N + 16) * 4));
          let maxAbs = 0;
          for (let i = 0; i < expected.length; i++) {
            const error = Math.abs(actual[prefix + i] - expected[i]);
            maxAbs = Math.max(maxAbs, error);
            assert.ok(Number.isFinite(actual[prefix + i]) && error <= 2e-4 + 2e-5 * Math.abs(expected[i]),
              `${dtype}/${mode} M=${M} N=${N} K=${K} [${i}] ${actual[prefix + i]} vs ${expected[i]}`);
          }
          assert.ok(actual.subarray(0, prefix).every((x) => x === -12345), 'prefix untouched');
          assert.ok(actual.subarray(prefix + M * N).every((x) => x === -12345), 'tail untouched');
          const bits = new Uint32Array(actual.buffer);
          if (baselineBits && variant.includes('_columns')) {
            const baseline = new Float32Array(baselineBits.buffer);
            for (let i = 0; i < actual.length; i++) {
              assert.ok(Math.abs(actual[i] - baseline[i]) <= 2e-4 + 2e-5 * Math.abs(baseline[i]),
                `${variant}/${mode} changes projection beyond the frozen F32 reference tolerance at ${i}`);
            }
          } else if (baselineBits) assert.deepEqual(bits, baselineBits, `${variant}/${mode} changes projection or guard bits`);
          else baselineBits = bits;
          console.log(JSON.stringify({ dtype, variant, M, N, K, bank, mode, maxAbs, rowGroupingBitwiseParity: !variant.includes('_columns') }));
        }
      }
    }
    const oddK = blockValues * 3;
    const oddBlocks = createWeightBuffer(upload(packedFixture(dtype, 4, oddK)), dtype, 'row', [4, oddK]);
    await assert.rejects(runQ40Matmul(createTensor(upload(new Float32Array(oddK)), 'f32', [1, oddK]), oddBlocks, 1, 4, oddK,
      { outputDtype: 'f32', variant: `${dtype}_pair` }), /whole block pairs/);
  }
  assert.deepEqual(errors, []);
  console.log('matmul-gguf-pair-gpu.test: ok');
} finally {
  for (const buffer of allocated) releaseBuffer(buffer);
  destroyDevice();
}
