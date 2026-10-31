import assert from 'node:assert/strict';
import { statSync } from 'node:fs';

const args = process.argv.slice(2);
const ordered = args[0] === '--ordered';
assert.ok(args.length === 0 || (ordered && args[1] === '--output-dir' && args.length === 3),
  'Usage: moe-grouped-physical.js [--ordered --output-dir <new-directory>]');

// This fixture is deliberately not a software-provider or ordinary unit test.
assert.equal(process.platform, 'linux', 'Physical fixture requires the authorized DRI host');
assert.ok(statSync('/dev/dri').isDirectory(), 'Physical /dev/dri access is required');
assert.equal(process.env.DOPPLER_GROUPED_LOCK_FD, '9', 'The guarded launcher must hold the timing lock');
const locked = statSync('/proc/self/fd/9');
assert.equal(`${locked.dev}:${locked.ino}`, process.env.DOPPLER_GPU_TIMING_LOCK_ID);

const { probeNodeGPU } = await import('../../tests/helpers/gpu-probe.js');
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { createCommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { runSoftmaxTopK, recordSoftmaxTopK } = await import('../../src/gpu/kernels/softmax.js');
const { groupedPrefillGeometry, executeGroupedSort } = await import('../../src/gpu/kernels/moe-grouped.js');
const { unifiedKernelWrapper } = await import('../../src/gpu/kernels/kernel-execution.js');
const { executeGroupedPrefill } = await import('../../src/inference/pipelines/text/moe/grouped-prefill.js');
const { runQwenSharedExpert } = await import('../../src/inference/pipelines/text/ffn/qwen-shared-expert.js');
const { runResidualAdd } = await import('../../src/gpu/kernels/residual.js');
const { createKernelRegistry, enterKernelRegistry } = await import('../../src/gpu/kernels/kernel-configs.js');
const { MoERouter } = await import('../../src/inference/moe-router.js');
const { runMoEFFNGPU } = await import('../../src/inference/pipelines/text/ffn/moe.js');
const { getRuntimeConfig } = await import('../../src/config/runtime.js');
const { float32ToFloat16 } = await import('../../src/formats/float16.js');

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, `Physical provider unavailable: ${probe.reason}`);
const device = getDevice();
const info = getKernelCapabilities().adapterInfo;
assert.ok(info && /amd|nvidia|intel|apple/i.test(JSON.stringify(info))
  && !/swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(info)), 'Physical adapter identity required');
if (ordered) {
  try {
    const { runOrderedGroupedPhysical } = await import('./moe-grouped-ordered-physical.js');
    const report = await runOrderedGroupedPhysical({ device, info, probe, outputDirectory: args[2] });
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await device.queue.onSubmittedWorkDone();
    destroyDevice();
  }
} else {
const errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const held = [];
const keep = b => { held.push(b); return b; };
const buffer = data => { const b = keep(acquireBuffer(Math.max(4, data.byteLength))); uploadData(b, data); return b; };
const tensor = (data, shape) => createTensor(buffer(data), 'f32', shape);
const dense = (data, shape) => createWeightBuffer(buffer(data), 'f32', 'row', shape);
const read = async (b, count) => new Float32Array(await readBuffer(b, count * 4));
const check = (actual, expected, label, tolerance = 0.0001) => {
  assert.equal(actual.length, expected.length);
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    assert.ok(Number.isFinite(actual[i]) && error <= tolerance, `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
    maxAbs = Math.max(maxAbs, error);
  }
  return maxAbs;
};
// Independent original-format encoder and dense oracle. No runtime decoder,
// grouped arithmetic, or Pooled generator is used to calculate expected values.
function packed(dtype, shape, salt) {
  const K = shape.at(-1), rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const bytes = { q4_0: 18, q4_1: 20, q8_0: 34 }[dtype];
  const raw = new Uint8Array(Math.ceil(rows * K / 32 * bytes / 4) * 4);
  const view = new DataView(raw.buffer), values = new Float32Array(rows * K);
  for (let row = 0; row < rows; row++) for (let block = 0; block < K / 32; block++) {
    const offset = (row * K / 32 + block) * bytes;
    const scale = ((row + block + salt) % 3 === 0 ? -1 : 1) / 64;
    const minimum = -0.0625 + ((row + block) % 3) / 64;
    view.setUint16(offset, float32ToFloat16(scale), true);
    if (dtype === 'q4_1') view.setUint16(offset + 2, float32ToFloat16(minimum), true);
    for (let k = 0; k < 32; k++) {
      const q = (row * 3 + block * 5 + k * 7 + salt) % 16;
      if (dtype === 'q8_0') view.setInt8(offset + 2 + k, q - 8);
      else raw[offset + (dtype === 'q4_1' ? 4 : 2) + k % 16] |= q << (k < 16 ? 0 : 4);
      values[row * K + block * 32 + k] = dtype === 'q4_1' ? scale * q + minimum : scale * (q - 8);
    }
  }
  return { weight: createWeightBuffer(buffer(raw), dtype, 'row', shape), values };
}
function dotRow(matrix, row, K, x) {
  let sum = 0;
  for (let k = 0; k < K; k++) sum += matrix[row * K + k] * x[k];
  return sum;
}
function routingReference(logits, T, E, topK, normalize) {
  const ids = [], weights = [];
  for (let t = 0; t < T; t++) {
    const row = Array.from(logits.subarray(t * E, (t + 1) * E));
    const selected = Array.from({ length: E }, (_, e) => e).sort((a, b) => row[b] - row[a] || a - b).slice(0, topK);
    const max = Math.max(...row), exp = row.map(x => Math.exp(x - max));
    const denominator = normalize ? selected.reduce((s, e) => s + exp[e], 0) : exp.reduce((a, b) => a + b, 0);
    ids.push(...selected); weights.push(...selected.map(e => exp[e] / denominator));
  }
  return { ids, weights };
}
const receipts = [];
try {
  const E = 256, topK = 8, H = 96, I = 96;
  const gate = packed('q4_0', [E, I, H], 1), up = packed('q4_0', [E, I, H], 4);
  const sharedGate = packed('q8_0', [32, H], 2), sharedUp = packed('q8_0', [32, H], 7);
  const sharedDown = packed('q4_0', [H, 32], 3);
  const sharedRouterValues = Float32Array.from({ length: H }, (_, k) => (k % 5 - 2) / 32);
  const sharedWeights = { sharedExpertGate: sharedGate.weight, sharedExpertUp: sharedUp.weight,
    sharedExpertDown: sharedDown.weight, sharedExpertRouter: dense(sharedRouterValues, [1, H]) };
  for (const dtype of ['q4_1', 'q4_0', 'q8_0']) {
    const down = packed(dtype, [E, H, I], 5);
    for (const T of [1, 7, 8, 9, 33]) {
      const normalize = T !== 7;
      const x = Float32Array.from({ length: T * H }, (_, i) => ((i * 7) % 23 - 11) / 32);
      const input = tensor(x, [T, H]);
      // Repeated expert 0, absent expert 255, ties, changing membership, and
      // >28-column placement rounds. Most experts have an empty group.
      const logits = Float32Array.from({ length: T * E }, (_, i) => {
        const t = Math.floor(i / E), e = i % E;
        const base = (t % 3) * 16;
        return e === 0 ? 4 : e > base && e <= base + 9 ? 2 : -4;
      });
      const logitBuffer = buffer(logits);
      const expectedRouting = routingReference(logits, T, E, topK, normalize);
      const geometry = { numTokens: T, numExperts: E, topK, hiddenSize: H, intermediateSize: I };
      const sizes = groupedPrefillGeometry(geometry, device.limits);
      for (const recorded of [false, true]) {
        const recorder = recorded ? createCommandRecorder('grouped-prefill-fixture') : null;
        const routes = recorder
          ? await recordSoftmaxTopK(recorder, logitBuffer, T, E, topK, { normalize, inputDtype: 'f32', weightsDtype: 'f32', modelType: 'transformer' })
          : await runSoftmaxTopK(logitBuffer, T, E, topK, { normalize, inputDtype: 'f32', weightsDtype: 'f32', modelType: 'transformer' });
        keep(routes.indices); keep(routes.weights);
        const groups = keep(acquireBuffer(sizes.groupBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
        const indirect = keep(acquireBuffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC));
        let routed;
        try {
          await executeGroupedSort(recorder, routes.indices, groups, indirect, geometry);
          routed = await executeGroupedPrefill(recorder, { inputTensor: input, indicesBuffer: routes.indices,
            weightsBuffer: routes.weights, weights: { numExperts: E, gate: gate.weight, up: up.weight, down: down.weight },
            ...geometry, swigluLimit: null });
          keep(routed.buffer);
          if (recorder) await recorder.submitAndWait();
          const ids = new Uint32Array(await readBuffer(routes.indices, T * topK * 4));
          assert.deepEqual([...ids], expectedRouting.ids, 'Router tie ordering remains ascending expert ID');
          check(await read(routes.weights, T * topK), expectedRouting.weights, 'router normalization', 0.00001);
          const schedule = new Uint32Array(await readBuffer(groups, sizes.groupBytes));
          const args = new Uint32Array(await readBuffer(indirect, 24));
          const seen = [];
          let chunk = 0, pair = 0;
          for (let e = 0; e < E; e++) {
            const selected = expectedRouting.ids.flatMap((id, index) => id === e ? [index] : []);
            for (let offset = 0; offset < selected.length; offset += 8) {
              const count = Math.min(8, selected.length - offset);
              assert.deepEqual([...schedule.slice(chunk * 4, chunk * 4 + 4)], [pair, count, e, 0]);
              const mapped = [...schedule.slice(sizes.pairOffset + pair, sizes.pairOffset + pair + count)];
              assert.deepEqual(mapped, selected.slice(offset, offset + count));
              seen.push(...mapped); pair += count; chunk++;
            }
          }
          assert.equal(new Set(seen).size, T * topK);
          assert.deepEqual([...args], [sizes.guGroups, chunk, 1, sizes.downGroups, chunk, 1]);
          const expected = new Float32Array(T * H);
          for (let t = 0; t < T; t++) {
            const xt = x.subarray(t * H, (t + 1) * H);
            for (let slot = 0; slot < topK; slot++) {
              const e = expectedRouting.ids[t * topK + slot];
              const activation = Float32Array.from({ length: I }, (_, r) => {
                const g = dotRow(gate.values, e * I + r, H, xt);
                return g / (1 + Math.exp(-Math.max(-15, Math.min(15, g)))) * dotRow(up.values, e * I + r, H, xt);
              });
              for (let h = 0; h < H; h++) expected[t * H + h] += expectedRouting.weights[t * topK + slot] * dotRow(down.values, e * H + h, I, activation);
            }
          }
          const maxAbs = check(await read(routed.buffer, T * H), expected, 'grouped down/scatter');
          // Existing shared expert uses different source formats and owns its gate.
          const combined = await runQwenSharedExpert(input, routed, sharedWeights,
            { numTokens: T, hiddenSize: H, intermediateSize: 32, kernelPath: null, phase: 'prefill', swigluLimit: null });
          keep(combined.buffer);
          const residual = await runResidualAdd(input, combined, T * H);
          keep(residual.buffer);
          for (let t = 0; t < T; t++) {
            const xt = x.subarray(t * H, (t + 1) * H);
            const a = Float32Array.from({ length: 32 }, (_, r) => {
              const g = dotRow(sharedGate.values, r, H, xt);
              return g / (1 + Math.exp(-Math.max(-15, Math.min(15, g)))) * dotRow(sharedUp.values, r, H, xt);
            });
            const router = dotRow(sharedRouterValues, 0, H, xt);
            const scale = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, router))));
            for (let h = 0; h < H; h++) expected[t * H + h] += scale * dotRow(sharedDown.values, h, 32, a) + xt[h];
          }
          check(await read(residual.buffer, T * H), expected, 'shared gate and one residual');
          receipts.push({ dtype, tokens: T, recorded, normalize, chunks: chunk, maxAbs });
        } finally { if (recorder) await recorder.abort(); }
      }
    }
  }
  // Exercise the actual FFN plumbing, including its shared-expert owner. A
  // candidate phase must not accidentally append the shared expert twice.
  const plumbingInput = tensor(Float32Array.from({ length: 9 * H }, (_, i) => (i % 11 - 5) / 32), [9, H]);
  const plumbingDown = packed('q4_1', [E, H, I], 9);
  const router = new MoERouter({ numExperts: E, topK, hiddenSize: H, normalizeWeights: true });
  router.loadWeights(dense(new Float32Array(E * H), [E, H]));
  const expertWeights = new Map();
  for (let e = 0; e < E; e++) expertWeights.set(`layer_0_expert_${e}`, {
    numExperts: E, expertIdx: e, expertFormat: 'qwen3_5',
    gate: gate.weight, up: up.weight, down: plumbingDown.weight,
  });
  const context = { config: { modelType: 'transformer', hiddenSize: H, intermediateSize: I,
    moeExpertIntermediateSize: I, moeSharedExpertIntermediateSize: 32, numExperts: E, moeTopK: topK,
    expertFormat: 'qwen3_5', hiddenActivation: 'silu', swigluLimit: null, rmsNormEps: 1e-6 },
    moeRouter: router, expertWeights, expertLoader: { loadExpert() { throw Error('Resident banks must not reload'); } },
    weights: new Map([['layer_0', sharedWeights]]), kernelPath: null, phase: 'prefill', executionPolicies: null,
    runtimeMoERoutingConfig: { ...getRuntimeConfig().inference.moe.routing, activeExpertSelection: 'topk-route' } };
  const baselineOutput = await runMoEFFNGPU(0, plumbingInput, 9, context);
  keep(baselineOutput.buffer);
  const candidateSteps = ['moe_grouped_sort', 'moe_grouped_gu', 'moe_grouped_down'].map(op => ({
    op, kernel: op === 'moe_grouped_sort' ? 'moe_grouped_sort.wgsl' : 'moe_grouped_matmul_f32.wgsl', entry: 'main',
  }));
  const candidateOutput = await runMoEFFNGPU(0, plumbingInput, 9, { ...context,
    kernelPath: { prefill: { steps: candidateSteps }, decode: { steps: [] } } });
  keep(candidateOutput.buffer);
  check(await read(candidateOutput.buffer, 9 * H), await read(baselineOutput.buffer, 9 * H), 'FFN integration');
  // Entirely empty sort writes zero indirect counts and never consumes indices.
  const emptyGroups = buffer(new Uint32Array(4)), emptyIndirect = keep(acquireBuffer(24,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC));
  await unifiedKernelWrapper('moe_grouped_sort', null, 'native_f32', [buffer(new Uint32Array(1)), emptyGroups, emptyIndirect],
    { num_routes: 0, num_experts: E, top_k: topK, pair_offset: 0, gu_groups: 1, down_groups: 1 }, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(await readBuffer(emptyIndirect, 24))], [1, 0, 1, 1, 0, 1]);
  // Recorded failure must retain submitted/pending resources until abort, then free them.
  const baseline = getBufferPool().getStats().activeBuffers;
  const recorder = createCommandRecorder('grouped-failure');
  const leave = enterKernelRegistry(createKernelRegistry({ validators: { moe_grouped_gu: { native_f32: {
    id: 'grouped-fixture-injected-failure', validate() { throw new Error('injected grouped GU failure'); },
  } } } }));
  const failureInput = tensor(new Float32Array(H), [1, H]);
  const ids = buffer(Uint32Array.from({ length: topK }, (_, i) => i)), routeWeights = buffer(new Float32Array(topK));
  const beforeFailure = getBufferPool().getStats().activeBuffers;
  try {
    await assert.rejects(executeGroupedPrefill(recorder, { inputTensor: failureInput, indicesBuffer: ids,
      weightsBuffer: routeWeights, weights: { numExperts: E, gate: gate.weight, up: up.weight, down: gate.weight },
      numTokens: 1, topK, hiddenSize: H, intermediateSize: I, swigluLimit: null }), /injected grouped GU failure/);
  } finally { leave(); await recorder.abort(); }
  assert.equal(getBufferPool().getStats().activeBuffers, beforeFailure, 'abort returns all owned scratch');
  assert.ok(beforeFailure >= baseline);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'pass', adapter: info, provider: probe.providerReceipt,
    scope: 'small physical fixtures only; full-model logit gate 0.001 and performance pending', receipts }, null, 2));
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const b of held) releaseBuffer(b);
  destroyDevice();
}

}
