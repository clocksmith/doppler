import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { selectTargetPlan } from '../../src/client/runtime/target-selector.js';
import { createTargetPlan } from '../../src/config/target-plan.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

// Signed fixture and injected device; these tests prove the authority boundary,
// while subgroup-portability-physical.test.js proves the affected operator math.
const predicate = { requiresF16: false, requiresSubgroups: true, minBufferSize: 4 };
const wgslSource = 'enable subgroups; requires subgroup_id; @compute @workgroup_size(1) fn main() {}';
const undeclared = await createSignedCapsuleFixture({ wgslSource, capabilityPredicate: predicate });
const declared = await createSignedCapsuleFixture({ wgslSource,
  capabilityPredicate: { ...predicate, requiredWgslFeatures: ['subgroup_id'] } });
const legacy = await createSignedCapsuleFixture();
const profile = { surface: 'test-webgpu', hasF16: false, hasSubgroups: true, maxBufferSize: 1024 };
let prepared = 0;
const device = { limits: { maxBufferSize: 1024 }, createBuffer: () => ({ destroy() {} }),
  createCommandEncoder() {}, queue: { writeBuffer() {} } };
function runtime(fixture, wgslLanguageFeatures) {
  return createDopplerRuntime({
    device: { getDevice: () => device, getProfile: () => ({ ...profile, wgslLanguageFeatures }) },
    artifactStore: fixture.artifactStore,
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    programFactory: async () => { prepared++; return { async close() {} }; },
  });
}
await assert.rejects(runtime(undeclared, ['subgroup_id']).openCapsule(undeclared.capsule), /TargetPlan.*language declarations/);
const implicit = await createSignedCapsuleFixture({ capabilityPredicate: predicate,
  wgslSource: 'enable subgroups; @compute @workgroup_size(1) fn main(@builtin(subgroup_id) id: u32) {}' });
await assert.rejects(runtime(implicit, ['subgroup_id']).openCapsule(implicit.capsule), /TargetPlan.*language declarations/);
await assert.rejects(runtime(declared, []).openCapsule(declared.capsule), /capability predicates/);
assert.equal(prepared, 0, 'missing language authority/support rejects before program preparation');
const session = await runtime(declared, ['subgroup_id']).openCapsule(declared.capsule);
await session.close();
const oldSession = await runtime(legacy, undefined).openCapsule(legacy.capsule);
await oldSession.close();
assert.equal(prepared, 2, 'old Capsules without new WGSL requirements retain their existing contract');
assert.equal(selectTargetPlan([declared.targetPlan, legacy.targetPlan], profile), legacy.targetPlan);
assert.equal(selectTargetPlan([declared.targetPlan, legacy.targetPlan], {
  ...profile, wgslLanguageFeatures: ['subgroup_id'],
}), declared.targetPlan);
for (const requiredWgslFeatures of [null, 'subgroup_id', ['bad-name'], ['subgroup_id', 'subgroup_id'], [1]]) {
  assert.throws(() => createTargetPlan({ ...declared.targetPlan,
    capabilityPredicate: { ...predicate, requiredWgslFeatures } }), /requiredWgslFeatures/);
}
console.log('capsule-wgsl-language.test: ok');
