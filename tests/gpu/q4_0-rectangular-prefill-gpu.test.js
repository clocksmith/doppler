import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture, decodePacked, projectReference } from '../helpers/q4_0-fixture.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { runQ40Matmul, recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`q4_0-rectangular-prefill-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const device = getDevice(), owned = [], errors = [], cases = [], skipped = new Set();
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = values => {
  const buffer = acquireBuffer(Math.ceil(values.byteLength / 4) * 4);
  owned.push(buffer); uploadData(buffer, values); return buffer;
};

try {
  for (const dtype of ['q4_0', 'q4_1']) {
    const variants = [dtype + '_prefill_tile64', dtype + '_prefill_tile64_bounded', dtype + '_prefill_tile128_bounded', ...(dtype === 'q4_0'
      ? ['q4_0_prefill_tile64x128', 'q4_0_prefill_tile128'] : ['q4_1_prefill_tile128'])];
    for (const [M, N, K] of [[3, 7, 64], [65, 67, 512], [129, 131, 512], [4, 31, 5120]]) {
      const packed = packedFixture(dtype, 2 * N, K);
      const values = Float32Array.from({ length: M * K + 4 }, (_, i) => Math.sin(i * 0.13) * 0.01);
      const input = createTensor(upload(values), 'f32', [values.length]);
      const weights = createWeightBuffer(upload(packed), dtype, 'row', [2, N, K]);
      const expected = projectReference(values.subarray(1), decodePacked(dtype, packed, 2 * N, K).subarray(K), M, N, K);
      let baseline = null;
      for (const variant of variants) {
        if (getKernelConfig('matmul', variant).sharedMemory > device.limits.maxComputeWorkgroupStorageSize) {
          skipped.add(variant); continue;
        }
        for (const recorded of [false, true]) {
          const output = upload(new Float32Array(M * N + 8).fill(-12345));
          const recorder = recorded ? createCommandRecorder('rectangular_reference') : null;
          const options = { variant, outputDtype: 'f32', outputBuffer: output, alpha: 0.5,
            aOffset: 4, bOffset: K / 32 * (dtype === 'q4_0' ? 18 : 20), cOffset: 16 };
          try {
            if (recorder) {
              await recordQ40Matmul(recorder, input, weights, M, N, K, options);
              await recorder.submitAndWait();
            } else await runQ40Matmul(input, weights, M, N, K, options);
          } finally { await recorder?.abort(); }
          const actual = new Float32Array(await readBuffer(output, (M * N + 8) * 4));
          let maxAbs = 0;
          for (let i = 0; i < M * N; i++) {
            const delta = Math.abs(actual[4 + i] - expected[i] * 0.5);
            maxAbs = Math.max(maxAbs, delta);
            assert.ok(Number.isFinite(delta) && delta <= 2e-4 + 2e-5 * Math.abs(expected[i]),
              `${variant}/${M}/${N}/${K}/${recorded} at ${i}`);
          }
          assert.ok(actual.subarray(0, 4).every(value => value === -12345));
          assert.ok(actual.subarray(M * N + 4).every(value => value === -12345));
          const bits = new Uint32Array(actual.buffer);
          if (baseline) assert.deepEqual(bits, baseline, 'Rectangular tiling changed native F32 output bits');
          else baseline = bits;
          cases.push({ dtype, M, N, K, variant, recorded, maxAbs, bitwiseNativeParity: true });
          if (variant !== variants[0]) {
            await assert.rejects(runQ40Matmul(input, weights, M, N, K,
              { ...options, outputBuffer: input.buffer }), /alias|undersized|requires/);
            await assert.rejects(runQ40Matmul(input, weights, M, N, K,
              { ...options, bOffset: 1 }), /offset/);
          }
        }
      }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, cases, skipped: [...skipped], errors,
    scope: 'Native Q4 rectangular prefill CPU reference, bit parity, offsets, tails and immediate/recorded execution' }));
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice(); await releaseNodeWebGPU();
}
