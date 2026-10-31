import assert from 'node:assert/strict';
import { validateSharedSlotGeometry, executeSharedSlotGateUp } from '../../src/gpu/kernels/moe-shared-slot.js';
import { resolveQwenSharedSlot } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
const geometry = { numTokens: 1, numExperts: 4, topK: 3, hiddenSize: 64, intermediateSize: 32, sharedSize: 96, swigluLimit: null };
validateSharedSlotGeometry(geometry);
for (const patch of [{ numTokens: 0 }, { topK: 5 }, { sharedSize: 33 }, { hiddenSize: 63 },
  { swigluLimit: undefined }, { swigluLimit: Infinity }]) {
  assert.throws(() => validateSharedSlotGeometry({ ...geometry, ...patch }));
}
const gu = { op: 'moe_shared_slot_gu', kernel: 'moe_shared_slot_gu_f32.wgsl', entry: 'main' };
const down = { op: 'moe_shared_slot_down', kernel: 'moe_shared_slot_down_f32.wgsl', entry: 'main' };
const kernelPath = { decode: { steps: [gu, down] }, prefill: { steps: [] }, layerOverrides: [
  { layers: [2], decode: { steps: [{ op: 'untouched' }] } },
] };
const lookup = { kernelPath, phase: 'decode', layerIdx: 0 };
assert.equal(resolveQwenSharedSlot(lookup), true);
assert.equal(resolveQwenSharedSlot({ ...lookup, phase: 'prefill' }), false);
assert.equal(resolveQwenSharedSlot({ ...lookup, layerIdx: 2 }), false);
for (const steps of [[gu], [down], [gu, gu, down], [gu, { ...down, kernel: 'residual.wgsl' }],
  [{ ...gu, constants: { TOP_K: 8 } }, down], [gu, down, { op: 'moe_shared_combine' }]]) {
  assert.throws(() => resolveQwenSharedSlot({ ...lookup, kernelPath: { decode: { steps } } }));
}
assert.throws(() => resolveQwenSharedSlot({ ...lookup, phase: undefined }));
// Fail before resource allocation when the device cannot hold the declared reduction.
const recorder = { device: { limits: { maxComputeInvocationsPerWorkgroup: 32, maxComputeWorkgroupStorageSize: 16384 } } };
await assert.rejects(executeSharedSlotGateUp(recorder, {}, {}, geometry), /workgroup limits/);
console.log('shared slot contract checks passed (CPU only)');
