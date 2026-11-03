import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture, decodePacked, projectReference } from '../helpers/q4_0-fixture.js';

// Native Q4_1 scale/min words, odd block counts, offsets and incomplete tiles.
// The new reduction order has a frozen F32 reference tolerance, not bit parity
// with scalar GEMV. Immediate and recorded execution must agree bit for bit.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`matmul-q41-columns-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
if (!getKernelCapabilities()?.hasSubgroups) {
  console.log('matmul-q41-columns-gpu.test: skipped (no subgroups)');
  destroyDevice();
  process.exit(0);
}
const { acquireBuffer, releaseBuffer, uploadData, readBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { createCommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { runQ40Matmul, recordQ40Matmul } = await import('../../src/gpu/kernels/q4_0.js');
const device = getDevice(), allocated = [], errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = data => {
  const bytes = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = acquireBuffer(bytes.byteLength);
  uploadData(buffer, bytes); allocated.push(buffer); return buffer;
};
try {
  for (const [M, N, K, bank] of [[1, 7, 32, 0], [3, 9, 96, 1], [4, 19, 512, 1],
    [5, 21, 2048, 0], [7, 33, 512, 1], [1, 130, 17408, 0]]) {
    const bytes = packedFixture('q4_1', (bank + 1) * N, K);
    const decoded = decodePacked('q4_1', bytes, (bank + 1) * N, K);
    const weight = createWeightBuffer(upload(bytes), 'q4_1', 'row', bank ? [2, N, K] : [N, K]);
    const values = Float32Array.from({ length: (M + bank) * K }, (_, i) => Math.sin(i * 0.23) * 0.125);
    const input = createTensor(upload(values), 'f32', [M + bank, K]);
    const expected = projectReference(values.subarray(bank * K), decoded.subarray(bank * N * K), M, N, K);
    for (const variant of ['q4_1_block_columns_single', 'q4_1_block_columns']) {
      let immediate = null;
      for (const mode of ['immediate', 'recorded']) {
        const recorder = mode === 'recorded' ? createCommandRecorder('q41_columns') : null;
        const prefix = bank * 4, output = upload(new Float32Array(M * N + 16).fill(-12345));
        const options = { outputDtype: 'f32', variant, outputBuffer: output, alpha: -0.5,
          aOffset: bank * K * 4, bOffset: bank * N * K / 32 * 20, cOffset: prefix * 4 };
        if (recorder) {
          await recordQ40Matmul(recorder, input, weight, M, N, K, options);
          await recorder.submitAndWait();
        } else await runQ40Matmul(input, weight, M, N, K, options);
        const actual = new Float32Array(await readBuffer(output, (M * N + 16) * 4));
        let maxAbs = 0;
        for (let i = 0; i < expected.length; i++) {
          const reference = -0.5 * expected[i], delta = Math.abs(actual[prefix + i] - reference);
          maxAbs = Math.max(maxAbs, delta);
          assert.ok(Number.isFinite(actual[prefix + i]) && delta <= 2e-4 + 2e-5 * Math.abs(reference),
            `${variant}/${mode} M=${M} N=${N} K=${K} at ${i}: ${actual[prefix + i]} vs ${reference}`);
        }
        assert.ok(actual.subarray(0, prefix).every(x => x === -12345));
        assert.ok(actual.subarray(prefix + M * N).every(x => x === -12345));
        const bits = new Uint32Array(actual.buffer);
        if (immediate) assert.deepEqual(bits, immediate); else immediate = bits;
        console.log(JSON.stringify({ variant, mode, M, N, K, bank, maxAbs }));
      }
    }
  }
  const input = createTensor(upload(new Float32Array(128)), 'f32', [2, 64]);
  const weight = createWeightBuffer(upload(packedFixture('q4_1', 4, 64)), 'q4_1', 'row', [4, 64]);
  await assert.rejects(runQ40Matmul(input, weight, 1, 4, 64,
    { outputDtype: 'f32', variant: 'q4_1_block_columns', aOffset: 4 }), /16-byte activations/);
  assert.deepEqual(errors, []);
  console.log('matmul-q41-columns-gpu.test: ok');
} finally {
  for (const buffer of allocated) releaseBuffer(buffer);
  destroyDevice();
}
