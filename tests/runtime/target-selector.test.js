import assert from 'node:assert/strict';
import { selectTargetPlan } from '../../src/client/runtime/target-selector.js';
import { createSignedPackFixture } from '../helpers/pack-v2-fixture.js';

const { targetPlan } = await createSignedPackFixture();
assert.equal(
  selectTargetPlan([targetPlan], { surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }).targetId,
  targetPlan.targetId
);
assert.throws(
  () => selectTargetPlan([{ ...targetPlan, capabilityPredicate: { ...targetPlan.capabilityPredicate, requiresF16: true } }], {
    surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024,
  }),
  /does not satisfy capability predicates and surface qualification/
);
assert.throws(
  () => selectTargetPlan([targetPlan], { surface: 'browser-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }),
  /surface qualification/
);
assert.throws(() => selectTargetPlan([], { surface: 'test-webgpu' }), /contains no target plans/);

console.log('✔ target-selector.test.js passed');
