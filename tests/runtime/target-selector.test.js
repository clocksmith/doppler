import assert from 'node:assert/strict';
import { selectTargetPlan } from '../../src/client/runtime/target-selector.js';
import { createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';

const { targetPlan } = await createSignedCapsuleFixture();
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

const profile = { surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 };
const alternative = { ...targetPlan, targetId: 'approved-alternative' };
const plans = [targetPlan, alternative];
const accepted = hashTargetPlan(alternative);
assert.equal(selectTargetPlan(plans, profile, { acceptedTargetPlanDigests: [accepted] }), alternative);
assert.throws(() => selectTargetPlan(plans, profile, { acceptedTargetPlanDigests: [] }), /not accepted/);

const rerank = (await createSignedCapsuleFixture({ operation: 'rerank' })).targetPlan;
assert.equal(selectTargetPlan([targetPlan, rerank], profile, { requiredOperations: ['rerank'] }), rerank);
assert.equal(selectTargetPlan([rerank, targetPlan], profile, { requiredOperations: ['generate'] }), targetPlan,
  'legacy qualification records without operation still mean generation');
assert.throws(() => selectTargetPlan([targetPlan, rerank], profile, { requiredOperations: ['generate', 'rerank'] }), /required operations/);
const both = { ...alternative, qualification: [...targetPlan.qualification, ...rerank.qualification] };
assert.equal(selectTargetPlan([rerank, both], profile, { requiredOperations: ['generate', 'rerank'] }), both);
assert.throws(() => selectTargetPlan([targetPlan], profile, { requiredOperations: ['unknown'] }), /required operations/);
assert.throws(() => selectTargetPlan([rerank], { ...profile, surface: 'other-host' }, { requiredOperations: ['rerank'] }), /surface qualification/);

assert.equal(selectTargetPlan(plans, profile, { preferredTargetPlanDigests: [accepted] }), alternative);
assert.equal(selectTargetPlan(plans, profile, {
  acceptedTargetPlanDigests: [hashTargetPlan(targetPlan)], preferredTargetPlanDigests: [accepted],
}), targetPlan, 'preference never grants application authorization');
const incompatible = { ...alternative, capabilityPredicate: { ...alternative.capabilityPredicate, requiresF16: true } };
assert.equal(selectTargetPlan([incompatible, targetPlan], profile, {
  preferredTargetPlanDigests: [hashTargetPlan(incompatible)],
}), targetPlan, 'preference cannot waive device capabilities');
assert.equal(selectTargetPlan([targetPlan, rerank], profile, {
  preferredTargetPlanDigests: [hashTargetPlan(targetPlan)], requiredOperations: ['rerank'],
}), rerank, 'preference cannot waive required operation qualification');
assert.equal(selectTargetPlan(plans, profile), targetPlan, 'signed Capsule order remains the default tie-break');
assert.equal(selectTargetPlan([alternative, targetPlan], profile), alternative);
assert.deepEqual(plans, [targetPlan, alternative], 'selection cannot mutate signed plan order');

for (const key of ['acceptedTargetPlanDigests', 'preferredTargetPlanDigests']) {
  for (const value of [null, 'digest', [null], ['invalid'], [accepted, accepted]]) {
    assert.throws(() => selectTargetPlan(plans, profile, { [key]: value }), new RegExp(key));
  }
}
for (const value of [null, 'rerank', [null], [''], ['rerank', 'rerank']]) {
  assert.throws(() => selectTargetPlan(plans, profile, { requiredOperations: value }), /requiredOperations/);
}
assert.throws(() => selectTargetPlan(plans, profile, { requiredOperation: 'rerank' }), /selection policy/);

console.log('✔ target-selector.test.js passed');
