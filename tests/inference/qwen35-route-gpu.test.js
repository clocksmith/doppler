import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer, isBufferActive } from '../../src/memory/buffer-pool.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';
import { dequantizeQ4KMRowWise } from '../../src/formats/q4k-dequant.js';
import { runRouteQ4Matmul, recordRouteQ4Matmul, runRouteScatter, recordRouteScatter } from '../../src/gpu/kernels/moe-route.js';
import { runQwenRouteExperts } from '../../src/inference/pipelines/text/moe/qwen-route-executor.js';
import { moeFeedForwardGPU } from '../../src/inference/pipelines/text/moe-gpu.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical Qwen route verification unavailable: ${probe.reason}`);
const errors = [], device = getDevice();
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const buffers = [];
const upload = values => {
  const buffer = acquireBuffer(values.byteLength);
  buffers.push(buffer);
  uploadData(buffer, values);
  return buffer;
};
const bank = (values, shape) => {
  const { quantized } = quantizeToQ4KMRowWise(values, shape);
  return { weight: createWeightBuffer(upload(quantized), 'q4k', 'row', shape),
    reference: dequantizeQ4KMRowWise(quantized, shape) };
};
const check = (actual, expected, label) => {
  let maxAbs = 0;
  for (let i = 0; i < expected.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    maxAbs = Math.max(maxAbs, error);
    assert.ok(Number.isFinite(actual[i]) && error <= 0.0001 + 0.00001 * Math.abs(expected[i]),
      `${label}[${i}]: ${actual[i]} versus ${expected[i]}`);
  }
  return maxAbs;
};
const receipts = [];
try {
  for (const [K, N, E] of [[1, 1, 3], [255, 7, 3], [256, 33, 3], [257, 37, 3], [512, 17, 3], [2048, 33, 3], [256, 5, 257]]) {
    const T = 3, topK = 2, routes = new Uint32Array([E - 1, 0, 1, E - 1, 0, 0]);
    const routing = new Float32Array([0.25, 0.75, 0.6, 0.4, 0.5, 0.5]);
    const input = Float32Array.from({ length: T * K }, (_, i) => Math.sin(i * 0.13) * 0.1);
    const weights = bank(Float32Array.from({ length: E * N * K }, (_, i) => Math.sin(i * 0.07) * 0.2), [E, N, K]);
    const indices = upload(routes), routingBuffer = upload(routing);
    for (const mode of ['token', 'route']) {
      const values = mode === 'token' ? input : Float32Array.from({ length: T * topK * K }, (_, i) => Math.cos(i * 0.09) * 0.1);
      const tensor = createTensor(upload(values), 'f32', [mode === 'token' ? T : T * topK, K]);
      const expected = Float64Array.from({ length: T * topK * N }, (_, index) => {
        const route = Math.floor(index / N), col = index % N;
        const row = mode === 'token' ? Math.floor(route / topK) : route;
        let sum = 0;
        for (let k = 0; k < K; k++) sum += values[row * K + k] * weights.reference[(routes[route] * N + col) * K + k];
        return sum;
      });
      const expectedCombined = Float64Array.from({ length: T * N }, (_, index) => {
        const t = Math.floor(index / N), col = index % N;
        let sum = 0;
        for (let k = 0; k < topK; k++) sum += routing[t * topK + k] * expected[(t * topK + k) * N + col];
        return sum;
      });
      for (const recorded of [false, true]) {
        const recorder = recorded ? createCommandRecorder('qwen-route-verification') : null;
        let result, combined;
        try {
          const options = { numRoutes: T * topK, numExperts: E, topK, N, K, inputMode: mode, alpha: 1 };
          result = recorder ? await recordRouteQ4Matmul(recorder, tensor, indices, weights.weight, options)
            : await runRouteQ4Matmul(tensor, indices, weights.weight, options);
          const scatter = { numTokens: T, hiddenSize: N, topK };
          combined = recorder ? await recordRouteScatter(recorder, result, routingBuffer, scatter)
            : await runRouteScatter(result, routingBuffer, scatter);
          if (recorder) await recorder.submitAndWait();
          const maxAbs = check(new Float32Array(await readBuffer(result.buffer, expected.length * 4)), expected, 'route');
          check(new Float32Array(await readBuffer(combined.buffer, expectedCombined.length * 4)), expectedCombined, 'combine');
          receipts.push({ K, N, E, mode, recorded, maxAbs });
        } finally {
          if (result) releaseBuffer(result.buffer);
          if (combined) releaseBuffer(combined.buffer);
          recorder?.abort();
        }
      }
    }
  }
  const T = 2, H = 256, I = 257, E = 3, topK = 2;
  const input = Float32Array.from({ length: T * H }, (_, i) => Math.cos(i * 0.1) * 0.1);
  const indices = new Uint32Array([2, 0, 1, 2]), routing = new Float32Array([0.25, 0.75, 0.6, 0.4]);
  const gateUp = bank(Float32Array.from({ length: E * 2 * I * H }, (_, i) => Math.sin(i * 0.03) * 0.1), [E, 2 * I, H]);
  const down = bank(Float32Array.from({ length: E * H * I }, (_, i) => Math.cos(i * 0.07) * 0.1), [E, H, I]);
  const expected = new Float64Array(T * H);
  for (let t = 0; t < T; t++) for (let r = 0; r < topK; r++) {
    const expert = indices[t * topK + r], activated = new Float64Array(I);
    for (let i = 0; i < I; i++) {
      let g = 0, up = 0;
      for (let k = 0; k < H; k++) {
        g += input[t * H + k] * gateUp.reference[(expert * 2 * I + i) * H + k];
        up += input[t * H + k] * gateUp.reference[(expert * 2 * I + I + i) * H + k];
      }
      activated[i] = g / (1 + Math.exp(-Math.max(-15, Math.min(15, g)))) * up;
    }
    for (let h = 0; h < H; h++) for (let i = 0; i < I; i++) {
      expected[t * H + h] += routing[t * topK + r] * activated[i] * down.reference[(expert * H + h) * I + i];
    }
  }
  const output = await runQwenRouteExperts({ inputTensor: createTensor(upload(input), 'f32', [T, H]),
    indicesBuffer: upload(indices), weightsBuffer: upload(routing), weights: { numExperts: E,
      expertFormat: 'qwen3_5', gateUp: gateUp.weight, down: down.weight }, expectedExpertFormat: 'qwen3_5',
    numTokens: T, topK, hiddenSize: H, intermediateSize: I, activationDtype: 'f32', hiddenActivation: 'silu', swigluLimit: null });
  try { receipts.push({ executor: true, maxAbs: check(new Float32Array(await readBuffer(output.buffer, expected.length * 4)), expected, 'executor') }); }
  finally { releaseBuffer(output.buffer); }
  const previous = getRuntimeConfig();
  setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'topk-route' } } } });
  try {
    const coreInput = upload(input), logits = new Float32Array([
      Math.log(0.75), -20, Math.log(0.25), -20, Math.log(0.6), Math.log(0.4),
    ]);
    const router = { gateWeight: {}, normalizeWeights: true, topK, lastLogitsDtype: 'f32',
      async computeRouterLogitsGPU() {
        const buffer = acquireBuffer(logits.byteLength);
        uploadData(buffer, logits);
        return buffer;
      } };
    const expertWeights = new Map([['layer_0_expert_0', { numExperts: E, expertFormat: 'qwen3_5',
      gateUp: gateUp.weight, down: down.weight }]]);
    const result = await moeFeedForwardGPU(coreInput, T, { modelType: 'transformer', hiddenSize: H,
      intermediateSize: I, expertIntermediateSize: I, numExperts: E, moeTopK: topK,
      expertFormat: 'qwen3_5', hiddenActivation: 'silu', swigluLimit: null, activationDtype: 'f32', kernelPath: null },
    router, expertWeights, { loadExpert() { throw new Error('Resident bank must not reload individual experts'); } }, 0, null);
    try { receipts.push({ coreWithFixtureRouter: true,
      maxAbs: check(new Float32Array(await readBuffer(result, expected.length * 4)), expected, 'moe core') }); }
    finally { releaseBuffer(result); }
  } finally { setRuntimeConfig(previous); }
  let failedOutput;
  const registry = createKernelRegistry({ validators: { moe_route_matmul: { q4k_f32: {
    id: 'test:route-output-rollback', validate({ bindings }) {
      failedOutput = bindings[2];
      throw new Error('Injected route dispatch validation failure');
    },
  } } } });
  const leave = enterKernelRegistry(registry);
  try {
    const tensor = createTensor(upload(input), 'f32', [T, H]), routeIds = upload(indices);
    const opts = { numRoutes: T * topK, numExperts: E, topK, N: 2 * I, K: H, inputMode: 'token', alpha: 1 };
    await assert.rejects(runRouteQ4Matmul(tensor, routeIds, gateUp.weight, opts), /Injected route/);
    assert.equal(isBufferActive(failedOutput), false, 'Failed immediate output returns to its owner');
    const borrowed = acquireBuffer(T * topK * 2 * I * 4);
    buffers.push(borrowed);
    await assert.rejects(runRouteQ4Matmul(tensor, routeIds, gateUp.weight, { ...opts, outputBuffer: borrowed }), /Injected route/);
    assert.equal(isBufferActive(borrowed), true, 'Borrowed output remains caller-owned on failure');
    const recorder = createCommandRecorder('qwen-route-rollback');
    await assert.rejects(recordRouteQ4Matmul(recorder, tensor, routeIds, gateUp.weight, opts), /Injected route/);
    assert.equal(isBufferActive(failedOutput), true, 'Failed recorded output retained until recorder cleanup');
    await recorder.abort();
    assert.equal(isBufferActive(failedOutput), false, 'Aborting recorder releases failed recorded output');
    receipts.push({ outputFailureOwnership: true });
  } finally { leave(); }
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ test: 'qwen35-route', passed: true, provider: probe.providerReceipt, cases: receipts }));
} finally {
  for (const buffer of buffers) releaseBuffer(buffer);
  destroyDevice();
}
