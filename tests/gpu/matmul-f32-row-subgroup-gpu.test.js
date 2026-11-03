import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { observeKernelDispatches } from '../helpers/kernel-dispatch-observer.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';

assert.ok((await probeNodeGPU({ installFileFetchShim: true })).ready);
const device = getDevice(), errors = [], cases = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const step = { op: 'q_proj', kernel: 'matmul_f32_row_subgroup.wgsl', entry: 'main',
  constants: { WORKGROUP_SIZE: 128, COLS_PER_WG: 4 } };
const kernelPath = { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };

try {
  for (const [M, N, K, transpose] of [[1, 1, 1, true], [3, 7, 35, true], [4, 48, 5120, true],
    [5, 17, 513, true], [4, 7, 65, false], [7, 5, 129, false]]) {
    const owned = [];
    const upload = values => {
      const buffer = acquireBuffer(values.byteLength); owned.push(buffer); uploadData(buffer, values); return buffer;
    };
    const inputs = Float32Array.from({ length: M * K }, (_, index) => Math.sin(index * 0.17) * 0.125);
    const weights = Float32Array.from({ length: N * K }, (_, index) => Math.cos(index * 0.03) * 0.0625);
    const alpha = -0.75, expected = new Float64Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += inputs[m * K + k] * weights[transpose ? n * K + k : k * N + n];
      expected[m * N + n] = sum * alpha;
    }
    try {
      const A = createTensor(upload(inputs), 'f32', [M, K]);
      const B = createWeightBuffer(upload(weights), 'f32', transpose ? 'row' : 'column', transpose ? [N, K] : [K, N]);
      let baseline;
      for (const recorded of [false, true]) {
        const outputBuffer = upload(new Float32Array(M * N + 128).fill(-12345));
        const options = { role: 'q_proj', outputDtype: 'f32', transposeB: transpose, alpha,
          kernelPath, outputBuffer, cOffset: 256 };
        const observation = await observeKernelDispatches(async () => {
          const recorder = recorded ? createCommandRecorder('f32-row-reference') : null;
          try {
            const output = recorder ? await recordMatmul(recorder, A, B, M, N, K, options)
              : await runMatmul(A, B, M, N, K, options);
            if (recorder) await recorder.submitAndWait();
            return output;
          } finally { await recorder?.abort(); }
        });
        assert.equal(observation.dispatches.length, 1);
        assert.equal(observation.dispatches[0].pipeline?.variant, 'f32_row_subgroup');
        assert.equal(observation.dispatches[0].pipeline.shaderFile, step.kernel);
        const values = new Float32Array(await readBuffer(outputBuffer, (M * N + 128) * 4));
        assert.ok(values.subarray(0, 64).every(value => value === -12345), 'output prefix overwritten');
        assert.ok(values.subarray(64 + M * N).every(value => value === -12345), 'output tail overwritten');
        const actual = values.slice(64, 64 + M * N);
        let maxAbs = 0;
        for (let index = 0; index < actual.length; index++) {
          const delta = Math.abs(actual[index] - expected[index]); maxAbs = Math.max(maxAbs, delta);
          assert.ok(Number.isFinite(actual[index]) && delta <= 1e-6 + 1e-5 * Math.abs(expected[index]),
            `reference mismatch ${M}/${N}/${K}/${transpose} at ${index}: ${delta}`);
        }
        if (baseline) assert.deepEqual(actual, baseline, 'immediate and recorded results differ');
        else baseline = actual;
        cases.push({ M, N, K, transpose, recorded, maxAbs, dispatch: observation.dispatches[0] });
      }
    } finally {
      await device.queue.onSubmittedWorkDone();
      for (const buffer of owned) releaseBuffer(buffer);
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, scope: 'F32 row-batched projection operator only', cases, errors }));
} finally { destroyDevice(); }
