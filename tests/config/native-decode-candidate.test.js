import assert from 'node:assert/strict';
import { buildNativeDecodeCandidate } from '../../tools/experiments/native-decode-candidate.js';
import { getLayerSteps } from '../../src/config/kernel-path-loader.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { resolveQwenSharedSlot } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
const path = { activationDtype: 'f32', kvDtype: 'f32', decode: { steps: [] }, prefill: { steps: [] },
  postLayer: [{ op: 'lm_head', kernel: 'matmul_q4_0.wgsl', entry: 'gemv', precision: { weightDtype: 'q6k' } }],
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
  assert.equal(steps.filter(s => s.op === 'residual_add').length, 1);
  assert.equal(resolveQwenSharedSlot({ kernelPath: candidate.kernelPath, phase: 'decode', layerIdx: i }), true);
}
assert.throws(() => buildNativeDecodeCandidate(path, 40, [{ ...selections[0], variant: 'q8_0_cooperative' }], []), /storage/);
assert.throws(() => buildNativeDecodeCandidate(path, 40, [selections[0], selections[0]], []), /Duplicate/);
assert.throws(() => buildNativeDecodeCandidate(path, 40, [], [40]), /valid indices/);
console.log('native candidate preserves all effective layers, prefill, precision and residuals (CPU only)');
