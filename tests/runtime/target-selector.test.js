import assert from 'node:assert/strict';
import { selectTargetPlan } from '../../src/client/runtime/target-selector.js';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createTargetPlan } from '../../src/config/target-plan.js';

const dummyKernel = { id: 'k1', file: 'k1.wgsl', entry: 'main', digest: `sha256:${'0'.repeat(64)}` };

const f16Subgroups = createTargetPlan({
  targetId: 'webgpu-f16-subgroups',
  modelId: 'demo-model',
  capabilityPredicate: { requiresF16: true, requiresSubgroups: true, minBufferSize: 0 },
  dtypes: { activation: 'f16-subgroups', kv: 'f16', weight: 'q4k' },
  kernelClosure: [dummyKernel],
});

const f16Standard = createTargetPlan({
  targetId: 'webgpu-f16',
  modelId: 'demo-model',
  capabilityPredicate: { requiresF16: true, requiresSubgroups: false, minBufferSize: 0 },
  dtypes: { activation: 'f16', kv: 'f16', weight: 'q4k' },
  kernelClosure: [dummyKernel],
});

const f32Safe = createTargetPlan({
  targetId: 'webgpu-f32-safe',
  modelId: 'demo-model',
  capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: 0 },
  dtypes: { activation: 'f32', kv: 'f32', weight: 'f32' },
  kernelClosure: [dummyKernel],
});

const targets = [f16Subgroups, f16Standard, f32Safe];

// Case 1: High end GPU (F16 + Subgroups) -> picks f16Subgroups
const highEndSelected = selectTargetPlan(targets, { hasF16: true, hasSubgroups: true });
assert.equal(highEndSelected.targetId, 'webgpu-f16-subgroups');

// Case 2: Standard mobile GPU (F16, no Subgroups) -> picks f16Standard
const mobileSelected = selectTargetPlan(targets, { hasF16: true, hasSubgroups: false });
assert.equal(mobileSelected.targetId, 'webgpu-f16');

// Case 3: CPU / Fallback (No F16) -> picks f32Safe
const safeSelected = selectTargetPlan(targets, { hasF16: false, hasSubgroups: false });
assert.equal(safeSelected.targetId, 'webgpu-f32-safe');

// Case 4: No matching target throws actionable error
assert.throws(
  () => selectTargetPlan([f16Subgroups], { hasF16: false, hasSubgroups: false }),
  /TargetSelector: Device does not satisfy capability predicates/
);

// Case 5: Composition Root runtime test
const runtime = createDopplerRuntime({
  device: { hasF16: true, hasSubgroups: false },
});

const session = await runtime.openPack({
  modelId: 'demo-model',
  targetPlans: targets,
});
assert.equal(session.selectedTargetId, 'webgpu-f16');

console.log('✔ target-selector.test.js passed');
