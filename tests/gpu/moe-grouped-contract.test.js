import assert from 'node:assert/strict';
import { groupedPrefillGeometry, executeGroupedProjection } from '../../src/gpu/kernels/moe-grouped.js';
import { resolveGroupedPrefill } from '../../src/inference/pipelines/text/moe/grouped-prefill.js';
import { buildInlineKernelPath } from '../../src/inference/pipelines/text/execution-runtime-builders.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

const limits = { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxBufferSize: 256 * 1024 * 1024 };
const shape = { numTokens: 33, numExperts: 256, topK: 8, hiddenSize: 96, intermediateSize: 32 };
const size = groupedPrefillGeometry(shape, limits);
assert.equal(size.numRoutes, 264);
assert.ok(size.maxChunks >= 8 * Math.ceil(33 / 8));
assert.ok(size.groupBytes >= (size.pairOffset + 264) * 4);
for (const patch of [{ numTokens: 0 }, { topK: 257 }, { numExperts: 257 }, { hiddenSize: 33 },
  { intermediateSize: 31 }, { numTokens: 1e15 }]) {
  assert.throws(() => groupedPrefillGeometry({ ...shape, ...patch }, limits), /Unsupported candidate/);
}
for (const patch of [{ maxComputeInvocationsPerWorkgroup: 128 }, { maxComputeWorkgroupSizeX: 128 },
  { maxComputeWorkgroupStorageSize: 8192 }, { maxStorageBuffersPerShaderStage: 4 },
  { maxStorageBufferBindingSize: 1024 }, { maxBufferSize: 1024 }, { maxComputeWorkgroupsPerDimension: 1 }]) {
  assert.throws(() => groupedPrefillGeometry(shape, { ...limits, ...patch }), /no automatic fallback/);
}
const operations = ['moe_grouped_sort', 'moe_grouped_gu', 'moe_grouped_down'];
const steps = operations.map(op => {
  const kernel = op === 'moe_grouped_sort' ? 'moe_grouped_sort.wgsl' : 'moe_grouped_matmul_f32.wgsl';
  return { section: 'layer', phase: 'prefill', layers: 'all', op, kernel, entry: 'main',
    kernelRef: { id: op, version: '1', digest: KERNEL_REF_CONTENT_DIGESTS[`${kernel}#main`] },
    src: 'hidden', dst: 'hidden', precision: { activationDtype: 'f32', outputDtype: 'f32', accumDtype: 'f32', mathDtype: 'f32' } };
});
const session = { compute: { defaults: { activationDtype: 'f32' } }, kvcache: { kvDtype: 'f16' } };
const kernelPath = buildInlineKernelPath([...steps, { section: 'layer', phase: 'decode', layers: 'all', op: 'moe_shared_slot_gu', kernel: 'moe_shared_slot_gu_f32.wgsl', entry: 'main' }], session, 'grouped-prefill-contract', 2);
const lookup = { kernelPath, phase: 'prefill', layerIdx: 0 };
assert.equal(resolveGroupedPrefill(lookup), true);
assert.equal(resolveGroupedPrefill({ ...lookup, phase: 'decode' }), false);
assert.equal(resolveGroupedPrefill({ ...lookup, kernelPath: null }), false);
assert.equal(resolveGroupedPrefill({ ...lookup, kernelPath: { prefill: { steps: [] } } }), false);
assert.equal(resolveGroupedPrefill({ ...lookup, layerIdx: 1, kernelPath: {
  prefill: { steps }, layerOverrides: [{ layers: [1], prefill: { steps: [{ op: 'unchanged' }] } }],
} }), false);
for (const invalid of [steps.slice(1), [...steps, steps[0]], [...steps].reverse(),
  steps.map((s, i) => i === 1 ? { ...s, kernel: 'residual.wgsl' } : s),
  steps.map((s, i) => i === 1 ? { ...s, constants: { GROUP_SIZE: 4 } } : s),
  [...steps, { op: 'moe_shared_slot_gu' }]]) {
  assert.throws(() => resolveGroupedPrefill({ ...lookup, kernelPath: { prefill: { steps: invalid } } }));
}
assert.throws(() => resolveGroupedPrefill({ ...lookup, phase: 'decode', kernelPath: { decode: { steps } } }), /cannot replace decode/);
await assert.rejects(executeGroupedProjection({ device: { limits: { ...limits, maxComputeWorkgroupSizeX: 128 } } },
  'moe_grouped_gu', {}, {}, null, {}, {}, shape), /no automatic fallback/);
console.log('moe-grouped-contract: CPU admission, explicit opt-in, phase/layer isolation passed');

const missingDecode = buildInlineKernelPath(steps, session, 'missing-decode', 2);
assert.throws(() => resolveGroupedPrefill({ kernelPath: missingDecode, phase: 'decode', layerIdx: 0 }), /cannot replace decode/);

// Declared output arithmetic must control both projection dispatch and the
// counting sort's indirect X counts. Otherwise ordered tiles leave rows unwritten.
const orderedSteps = steps.map(step => {
  if (step.op === 'moe_grouped_sort') return step;
  const config = getKernelConfig(step.op, 'native_f32_ordered');
  return { ...step, kernel: config.shaderFile, entry: config.entryPoint,
    kernelRef: { ...step.kernelRef,
      digest: KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`] } };
});
const orderedPath = { prefill: { steps: orderedSteps }, decode: { steps: [] } };
const orderedShape = { ...shape, hiddenSize: 2048, intermediateSize: 512,
  kernelPath: orderedPath, phase: 'prefill', layerIdx: 0 };
const orderedSize = groupedPrefillGeometry(orderedShape, limits);
assert.equal(orderedSize.guGroups, 64, '1024 output rows / 16 per workgroup');
assert.equal(orderedSize.downGroups, 64, '2048 output rows / 32 per workgroup');
assert.equal(resolveGroupedPrefill(orderedShape), true);
assert.deepEqual(groupedPrefillGeometry({ ...orderedShape, kernelPath: { prefill: { steps } } }, limits),
  groupedPrefillGeometry({ ...orderedShape, kernelPath: null }, limits), 'original declared and direct API stay equivalent');
assert.equal(groupedPrefillGeometry({ ...orderedShape, hiddenSize: 96, intermediateSize: 96 }, limits).downGroups, 3);
assert.deepEqual(groupedPrefillGeometry({ ...orderedShape, kernelPath: {
  ...orderedPath, layerOverrides: [{ layers: [1], prefill: { steps } }],
}, layerIdx: 1 }, limits), groupedPrefillGeometry({ ...orderedShape, kernelPath: null }, limits));
for (const patch of [{ phase: 'decode' }, { phase: undefined }, { layerIdx: undefined }, { layerIdx: -1 },
  { kernelPath: { prefill: { steps: orderedSteps.slice(0, 2) } } },
  { kernelPath: { prefill: { steps: [...orderedSteps, orderedSteps[1]] } } },
  { kernelPath: { prefill: { steps: orderedSteps.map(s => s.op === 'moe_grouped_gu' ? { ...s, constants: { REDUCTION_LANES: 16 } } : s) } } },
  { kernelPath: { prefill: { steps: [orderedSteps[0], orderedSteps[1], steps[2]] } } },
]) {
  assert.throws(() => groupedPrefillGeometry({ ...orderedShape, ...patch }, limits), /MoE grouped prefill/);
}
assert.throws(() => groupedPrefillGeometry(orderedShape, { ...limits, maxComputeWorkgroupsPerDimension: 63 }),
  /dispatch exceeds/);
console.log('moe-grouped-contract: ordered selection, full row coverage, layer scope and invalid program rejection passed');
