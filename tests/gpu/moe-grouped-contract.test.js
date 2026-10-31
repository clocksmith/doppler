import assert from 'node:assert/strict';
import { groupedPrefillGeometry, executeGroupedProjection } from '../../src/gpu/kernels/moe-grouped.js';
import { resolveGroupedPrefill } from '../../src/inference/pipelines/text/moe/grouped-prefill.js';
import { buildInlineKernelPath } from '../../src/inference/pipelines/text/execution-runtime-builders.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

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
