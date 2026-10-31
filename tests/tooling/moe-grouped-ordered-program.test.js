import assert from 'node:assert/strict';
import { buildOrderedGroupedProgram, orderedGroupedDeclarations, prepareOrderedGroupedCandidate }
  from '../../tools/experiments/moe-grouped-ordered-program.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { getLayerSteps, getKernelPathStepConfig } from '../../src/config/kernel-path-loader.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { createRDRRManifestFixture } from '../helpers/rdrr-manifest-fixture.js';

const norm = { op: 'input_norm', kernel: 'rmsnorm.wgsl', entry: 'main',
  digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['rmsnorm.wgsl#main']}` };
const unaffected = [norm,
  { ...norm, op: 'attention', marker: 'attention owner', constants: { TILE_SIZE: 16 } },
  { ...norm, op: 'moe_router_topk', marker: 'topk owner' },
  { ...norm, op: 'ffn_residual', marker: 'single outer residual' }];
const baseline = {
  id: 'synthetic-ordered-selection', name: 'Synthetic selection fixture', activationDtype: 'f32', kvDtype: 'f32',
  prefill: { steps: unaffected }, decode: { steps: [{ ...norm, marker: 'qualified decode' }] },
  layerOverrides: [
    { layers: [1], prefill: { steps: [{ ...norm, marker: 'explicit prefill' }, unaffected[3]] },
      decode: { steps: [{ ...norm, marker: 'explicit decode' }] } },
    { layers: [2], steps: [{ ...norm, marker: 'shared phase steps' }] },
    { layers: [3], prefill: { steps: [] }, decode: { steps: [{ ...norm, marker: 'decode only' }] } },
  ],
  postLayer: [{ ...norm, op: 'lm_head', marker: 'Q6 head owner' }],
  sampling: [{ ...norm, op: 'sample', marker: 'sampling owner' }],
};
const before = structuredClone(baseline);
const candidate = buildOrderedGroupedProgram(baseline, 4);
assert.deepEqual(baseline, before, 'Preparation must not mutate the accepted program');
assert.deepEqual(candidate.postLayer, baseline.postLayer);
assert.deepEqual(candidate.sampling, baseline.sampling);
assert.deepEqual(candidate.layerOverrides[2].prefill, { steps: [] }, 'Empty override keeps its inheritance');
for (let layer = 0; layer < 4; layer++) {
  assert.deepEqual(getLayerSteps(candidate, layer, 'decode'), getLayerSteps(baseline, layer, 'decode'));
  const steps = getLayerSteps(candidate, layer, 'prefill');
  assert.deepEqual(steps.filter(step => !step.op.startsWith('moe_grouped_')), getLayerSteps(baseline, layer, 'prefill'));
  for (const operation of ['moe_grouped_gu', 'moe_grouped_down']) {
    const selected = getKernelPathStepConfig(operation, steps.find(step => step.op === operation), 'prefill');
    assert.equal(selected.variant, 'native_f32_ordered', 'Production lookup must select ordered arithmetic');
  }
}
assert.deepEqual(buildOrderedGroupedProgram(candidate, 4), candidate, 'Explicit preparation is idempotent');

const oldGrouped = structuredClone(baseline);
const oldSteps = ['moe_grouped_sort', 'moe_grouped_gu', 'moe_grouped_down'].map(op => {
  const config = getKernelConfig(op, 'native_f32');
  return { op, kernel: config.shaderFile, entry: config.entryPoint,
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`]}` };
});
oldGrouped.prefill.steps.push(...oldSteps);
const replaced = buildOrderedGroupedProgram(oldGrouped, 4);
assert.deepEqual(replaced.prefill.steps.slice(-3, -2), oldGrouped.prefill.steps.slice(-3, -2),
  'Ordered arithmetic cannot change the existing sort contract');
assert.deepEqual(replaced.prefill.steps.slice(0, -3), oldGrouped.prefill.steps.slice(0, -3));
assert.equal(replaced.prefill.steps.filter(step => step.op === 'moe_grouped_gu').length, 1);
assert.notEqual(replaced.prefill.steps.at(-1).kernel, oldGrouped.prefill.steps.at(-1).kernel);

for (const [edit, error] of [
  [value => { value.activationDtype = 'f16'; }, /F32 activation/],
  [value => { value.decode.steps = []; }, /separately declared/],
  [value => { delete value.prefill; }, /separately declared/],
  [value => { value.layerOverrides[0].layers = [4]; }, /layer count/],
  [value => { value.prefill.steps.push(oldSteps[1]); }, /Declare exactly/],
  [value => { value.decode.steps.push(...oldSteps); }, /cannot replace decode/],
  [value => { value.prefill.steps.push({ ...norm, op: 'moe_shared_slot_gu' }); }, /shared-slot/],
]) {
  const invalid = structuredClone(baseline); edit(invalid);
  assert.throws(() => buildOrderedGroupedProgram(invalid, 4), error);
}
for (const count of [0, -1, 1.5, undefined]) assert.throws(() => buildOrderedGroupedProgram(baseline, count), /layer count/);

// Format fixture: the real parser and closure builder admit mechanism additions
// while semantic operations, session policy, tensors and old declarations stay exact.
const manifest = createRDRRManifestFixture();
manifest.architecture.numLayers = 4;
manifest.inference.schema = 'doppler.execution/v1';
manifest.inference.execution = {
  inlineKernelPath: false, kernels: { norm: { kernel: norm.kernel, entry: norm.entry, digest: norm.digest } },
  preLayer: [], decode: [['input_norm', 'norm']], prefill: [['input_norm', 'norm']], postLayer: [],
};
const runtimeConfig = { inference: { kernelPath: baseline, session: { arbitraryPinnedPolicy: true } } };
const manifestBefore = structuredClone(manifest);
const runtimeBefore = structuredClone(runtimeConfig);
const prepared = await prepareOrderedGroupedCandidate(manifest, runtimeConfig);
assert.deepEqual(manifest, manifestBefore);
assert.deepEqual(runtimeConfig, runtimeBefore);
assert.deepEqual(prepared.runtimeConfig.inference.session, runtimeConfig.inference.session);
const restoredManifest = structuredClone(prepared.manifest);
for (const op of Object.keys(orderedGroupedDeclarations())) delete restoredManifest.inference.execution.kernels[`${op}_ordered_candidate`];
delete restoredManifest.inference.execution.mechanismKernels;
assert.deepEqual(restoredManifest, manifestBefore, 'Only three explicit mechanism declarations may change');
assert(prepared.closure.modules.some(module => module.id === 'moe_grouped_gu_ordered_candidate'));
assert.equal(prepared.routing.length, 4);
await assert.rejects(prepareOrderedGroupedCandidate(manifest, { inference: {} }), /F32 activation/);

console.log('moe-grouped-ordered-program: production selection, phase isolation, parser and closure passed');
