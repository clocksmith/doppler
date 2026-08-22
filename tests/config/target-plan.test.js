import assert from 'node:assert/strict';
import {
  createTargetPlan,
  hashTargetPlan,
  matchesDeviceCapability,
  validateTargetPlan,
} from '../../src/config/target-plan.js';

const dummyKernel = {
  id: 'matmul_f16',
  file: 'matmul.wgsl',
  entry: 'main',
  digest: `sha256:${'a'.repeat(64)}`,
};

const plan = createTargetPlan({
  targetId: 'webgpu-f16-subgroups',
  modelId: 'test-qwen-model',
  capabilityPredicate: {
    requiresF16: true,
    requiresSubgroups: true,
    minBufferSize: 64 * 1024 * 1024,
  },
  dtypes: {
    activation: 'f16-subgroups',
    kv: 'f16',
    weight: 'q4k',
  },
  kernelClosure: [dummyKernel],
});

const validation = validateTargetPlan(plan);
assert.equal(validation.ok, true);

const hash = hashTargetPlan(plan);
assert.match(hash, /^sha256:[0-9a-f]{64}$/);

// Capability predicate matching checks
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: true, maxBufferSize: 128 * 1024 * 1024 }), true);
assert.equal(matchesDeviceCapability(plan, { hasF16: false, hasSubgroups: true, maxBufferSize: 128 * 1024 * 1024 }), false);
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: false, maxBufferSize: 128 * 1024 * 1024 }), false);
assert.equal(matchesDeviceCapability(plan, { hasF16: true, hasSubgroups: true, maxBufferSize: 16 * 1024 * 1024 }), false);

console.log('✔ target-plan.test.js passed');
