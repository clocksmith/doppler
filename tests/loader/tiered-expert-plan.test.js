import assert from 'node:assert/strict';
import { DEFAULT_EXPERT_CACHE_CONFIG, validateQwen35RoutedConfig } from '../../src/config/schema/runtime/loading.schema.js';
import { planTieredExpertPlacement } from '../../src/loader/experts/tiered-expert-plan.js';

const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
  storageMode: 'tiered-compact', tiered: { residentLayers: [0], parkedLayers: [1], mappedCacheBytes: 32, mappedProtectedShare: 0 },
  sourceReadMode: 'expert-lease-batch', cacheBytes: 256, gpuBudgetBytes: 1024, reserveBytes: 128 };
const plans = Array.from({ length: 3 }, (_, layer) => ({ layer, numLayers: 3, numExperts: 4,
  hiddenSize: 2, intermediateSize: 1, expertBytes: 12, bankBytes: 48,
  planes: ['gate', 'up', 'down'].map(role => ({ role, name: `layer_${layer}_${role}`, expertBytes: 4, bytes: 16 })) }));
const limits = { maxBufferSize: 64, maxStorageBufferBindingSize: 64 };
const plan = (change = {}, assigned = plans, device = limits, topK = 2) =>
  planTieredExpertPlacement(assigned, { ...policy, ...change }, device, topK);
const result = plan();
assert.equal(result.residentBytes, 48);
assert.equal(result.parkedBytes, 48);
assert.equal(result.mappedCacheBytes, 32);
assert.equal(result.compactBytes, 128);
assert.equal(result.cacheUpperBytes, 256);
assert.deepEqual([...result.layers.values()].map(row => [row.plan.layer, row.mode, row.capacity]),
  [[0, 'resident', 4], [1, 'parked', 4], [2, 'streamed', 4]]);
assert.deepEqual(result.compactLayerBytes, { 1: 64, 2: 64 });
const partition = plan({}, [plans[2]]);
assert.deepEqual([...partition.layers.keys()], [2], 'Partition plans preserve original layer indices');
assert.equal(partition.residentBytes, 0);
assert.equal(partition.parkedBytes, 0);
assert.equal(partition.cacheUpperBytes, 96);
assert.throws(() => plan({ cacheBytes: 128 }), /top-K/);
assert.throws(() => plan({ tiered: { ...policy.tiered, mappedCacheBytes: 256, mappedProtectedShare: 0 } }), /cache or device/);
assert.throws(() => plan({ compactLayerBytes: { 1: 96, 2: 160 } }), /remaining model cache/);
assert.throws(() => plan({ compactLayerBytes: { 0: 64, 2: 64 } }), /exactly the assigned nonresident/);
assert.throws(() => plan({}, plans, { ...limits, maxStorageBufferBindingSize: 8 }), /cache or device/);
assert.throws(() => plan({}, [plans[0], plans[0]]), /geometry/);
assert.throws(() => plan({}, [{ ...plans[0], expertBytes: 20 }]), /geometry/);
assert.throws(() => plan({}, plans, limits, 5), /geometry/);
for (const tiered of [null, {}, { ...policy.tiered, parkedLayers: [0] },
  { ...policy.tiered, mappedCacheBytes: -4 }, { ...policy.tiered, residentLayers: [0, 0] },
  { ...policy.tiered, extra: true }, { ...policy.tiered, mappedProtectedShare: undefined },
  { ...policy.tiered, mappedProtectedShare: -0.1 }, { ...policy.tiered, mappedProtectedShare: 1.1 },
  { ...policy.tiered, mappedProtectedShare: NaN }]) {
  assert.throws(() => validateQwen35RoutedConfig({ ...policy, tiered }), /Tiered experts/);
}
assert.throws(() => plan({ tiered: { ...policy.tiered, residentLayers: [3] } }), /geometry/);
assert.throws(() => validateQwen35RoutedConfig({ ...policy, sourceReadMode: 'plane-serial' }), /Tiered experts/);
assert.throws(() => validateQwen35RoutedConfig({ ...policy, compactPrefillBytes: 4 }), /Tiered experts/);
assert.throws(() => validateQwen35RoutedConfig({ ...policy, storageMode: 'compact-decode' }), /Tiered placement/);
assert.equal(validateQwen35RoutedConfig(structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed)).tiered, null);
console.log('tiered expert plan: explicit model-bound placement, partition indices, source extents, device/top-K and total-budget admission passed');
