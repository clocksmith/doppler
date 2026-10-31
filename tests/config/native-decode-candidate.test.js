import assert from 'node:assert/strict';
import { buildNativeDecodeCandidate } from '../../tools/experiments/native-decode-candidate.js';
import { getLayerSteps, getKernelPathStepConfig } from '../../src/config/kernel-path-loader.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { resolveQwenSharedSlot } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
const path = { activationDtype: 'f32', kvDtype: 'f32', decode: { steps: [] }, prefill: { steps: [] },
  postLayer: ['lm_head', 'lm_head_prefill'].map(op => ({ op, kernel: 'matmul_q4_0.wgsl', entry: 'gemv',
    weights: 'lm_head', constants: getKernelConfig('matmul', 'q6k_gemv').wgslOverrides,
    precision: { weightDtype: 'q6k' } })),
  layerOverrides: Array.from({ length: 40 }, (_, layerIdx) => {
    const dtype = layerIdx % 2 ? 'q8_0' : 'q4_0', config = getKernelConfig('matmul', `${dtype}_gemv`);
    return { layers: [layerIdx], decode: { steps: [{ op: 'q_proj', weights: `layer.${layerIdx}.q`,
      kernel: config.shaderFile, entry: config.entryPoint, constants: config.wgslOverrides }, { op: 'residual_add' }] },
      prefill: { steps: [{ op: `prefill_${layerIdx}` }] } };
  }),
};
const before = structuredClone(path);
const selections = Array.from({ length: 40 }, (_, layerIdx) => ({ layerIdx, op: 'q_proj',
  variant: `${layerIdx % 2 ? 'q8_0' : 'q4_0'}_cooperative` }));
const candidate = buildNativeDecodeCandidate(path, 40, selections, Array.from({ length: 40 }, (_, i) => i));
assert.deepEqual(path, before);
assert.equal(candidate.qualified, false);
assert.deepEqual(candidate.kernelPath.postLayer, path.postLayer);
for (let i = 0; i < 40; i++) {
  assert.deepEqual(getLayerSteps(candidate.kernelPath, i, 'prefill'), getLayerSteps(path, i, 'prefill'));
  const steps = getLayerSteps(candidate.kernelPath, i, 'decode');
  assert.equal(steps[0].weights, `layer.${i}.q`);
  assert.equal(steps[0].entry, 'cooperative');
  const resolved = getKernelPathStepConfig('matmul', steps[0], 'decode');
  assert.equal(resolved.weightDtype, i % 2 ? 'q8_0' : 'q4_0');
  assert.equal(resolved.variantMetadata.q40Geometry, 'cooperative');
  assert.equal(steps.filter(s => s.op === 'residual_add').length, 1);
  assert.equal(resolveQwenSharedSlot({ kernelPath: candidate.kernelPath, phase: 'decode', layerIdx: i }), true);
}
assert.throws(() => buildNativeDecodeCandidate(path, 40, [{ ...selections[0], variant: 'q8_0_cooperative' }], []), /storage/);
assert.throws(() => buildNativeDecodeCandidate(path, 40, [selections[0], selections[0]], []), /Duplicate/);
assert.throws(() => buildNativeDecodeCandidate(path, 40, [], [40]), /valid indices/);
const dtypes = ['q4_0', 'q4_1', 'q5_0', 'q8_0', 'q6k'];
const formats = structuredClone(path);
for (let i = 0; i < 40; i++) {
  formats.layerOverrides[i].decode.steps[0].constants = getKernelConfig('matmul', `${dtypes[i % 5]}_gemv`).wgslOverrides;
}
const head = { scope: 'postLayer', op: 'lm_head', variant: 'q6k_cooperative' };
const complete = buildNativeDecodeCandidate(formats, 40,
  [...Array.from({ length: 40 }, (_, layerIdx) => ({ layerIdx, op: 'q_proj', variant: `${dtypes[layerIdx % 5]}_cooperative` })), head],
  Array.from({ length: 40 }, (_, i) => i)).kernelPath;
for (let i = 0; i < 40; i++) {
  const step = getLayerSteps(complete, i, 'decode')[0];
  assert.equal(getKernelPathStepConfig('matmul', step, 'decode').weightDtype, dtypes[i % 5]);
  assert.deepEqual(getLayerSteps(complete, i, 'prefill'), getLayerSteps(formats, i, 'prefill'));
}
assert.equal(getKernelPathStepConfig('matmul', complete.postLayer[0], 'decode').weightDtype, 'q6k');
assert.equal(complete.postLayer[0].entry, 'cooperative');
assert.deepEqual(complete.postLayer[1], formats.postLayer[1]);
assert.throws(() => buildNativeDecodeCandidate(formats, 40, [head, head], []), /Duplicate/);
assert.throws(() => buildNativeDecodeCandidate(formats, 40, [{ ...head, variant: 'q8_0_cooperative' }], []), /storage/);
assert.throws(() => buildNativeDecodeCandidate({ ...formats, postLayer: [formats.postLayer[0]] }, 40, [head], []), /lm_head_prefill/);
console.log('native candidate preserves all effective layers, prefill, precision and residuals (CPU only)');
