import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { MoERouter } from '../../src/inference/moe-router.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { float32ToFloat16, float16ToFloat32 } from '../../src/formats/float16.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical router storage verification unavailable: ${probe.reason}`);
const device = getDevice(), errors = [], receipts = [], buffers = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
function upload(values) {
  const buffer = acquireBuffer(values.byteLength);
  buffers.push(buffer);
  uploadData(buffer, values);
  return buffer;
}
function compare(actual, expected) {
  let maxAbs = 0;
  for (let i = 0; i < expected.length; i++) {
    const difference = Math.abs(actual[i] - expected[i]);
    maxAbs = Math.max(maxAbs, difference);
    assert.ok(Number.isFinite(actual[i]) && difference <= 0.0001 + 0.00001 * Math.abs(expected[i]),
      `router[${i}]: ${actual[i]} versus ${expected[i]}`);
  }
  return maxAbs;
}
try {
  // 28 x 2048 into 256 experts matches the failing 35B arithmetic prefill geometry.
  // 1 and 64 rows also exercise decode and the tiled prefill selection boundary.
  for (const [M, N, K] of [[1, 256, 2048], [28, 256, 2048], [64, 256, 2048], [28, 1, 2048]]) {
    const input = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.13) * 0.1);
    const encoded = Uint16Array.from({ length: N * K }, (_, i) => float32ToFloat16(Math.cos(i * 0.07) * 0.1));
    const decoded = Float32Array.from(encoded, float16ToFloat32);
    const expected = Float64Array.from({ length: M * N }, (_, i) => {
      const row = Math.floor(i / N), column = i % N;
      let sum = 0;
      for (let k = 0; k < K; k++) sum += input[row * K + k] * decoded[column * K + k];
      return sum;
    });
    const tensor = createTensor(upload(input), 'f32', [M, K]);
    const weight = createWeightBuffer(upload(encoded), 'f16', 'row', [N, K]);
    for (const recorded of [false, true]) {
      const recorder = recorded ? createCommandRecorder('router-storage-verification') : null;
      let result;
      try {
        const options = { bDtype: 'f16', outputDtype: 'f32', preferF16: false, transposeB: true,
          role: N === 1 ? 'moe_shared_router' : 'moe_router' };
        result = recorder ? await recordMatmul(recorder, tensor, weight, M, N, K, options)
          : await runMatmul(tensor, weight, M, N, K, options);
        if (recorder) await recorder.submitAndWait();
        assert.equal(result.dtype, 'f32');
        receipts.push({ M, N, K, recorded,
          maxAbs: compare(new Float32Array(await readBuffer(result.buffer, M * N * 4)), expected) });
      } finally {
        recorder?.abort();
        if (result) releaseBuffer(result.buffer);
      }
    }
    const router = new MoERouter({ numExperts: N, hiddenSize: K, topK: Math.min(N, 8), normalizeWeights: true });
    router.loadWeights(weight);
    const output = await router.computeRouterLogitsGPU(tensor.buffer, M, null, { inputDtype: 'f32', outputDtype: 'f32' });
    try {
      assert.equal(router.lastLogitsDtype, 'f32');
      receipts.push({ M, N, K, consumer: 'MoERouter.computeRouterLogitsGPU',
        maxAbs: compare(new Float32Array(await readBuffer(output, M * N * 4)), expected) });
    } finally { releaseBuffer(output); }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', provider: probe.providerReceipt, receipts }, null, 2));
} finally {
  for (const buffer of buffers) releaseBuffer(buffer);
  destroyDevice();
}
