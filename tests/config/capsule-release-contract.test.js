import assert from 'node:assert/strict';
import { validateCapsuleReleaseContract } from '../../src/config/capsule-release-contract.js';
import { createCapsuleReleaseFixture } from '../helpers/capsule-v2-fixture.js';

const upgrade = createCapsuleReleaseFixture();
assert.equal(validateCapsuleReleaseContract(upgrade).ok, true);
const initial = structuredClone(upgrade);
initial.lifecycle.supersedes = null;
initial.lifecycle.migration = null;
initial.lifecycle.failedUpgrade.previousCapsuleId = null;
initial.lifecycle.failedUpgrade.previousSemanticRoot = null;
assert.deepEqual(validateCapsuleReleaseContract(initial), { ok: true, errors: [] });

for (const [name, source, mutate] of [
  ['missing predecessor', initial, (value) => { delete value.lifecycle.supersedes; }],
  ['invented predecessor ID', initial, (value) => { value.lifecycle.failedUpgrade.previousCapsuleId = 'invented'; }],
  ['invented predecessor root', initial, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = upgrade.lifecycle.supersedes.semanticRoot; }],
  ['missing predecessor ID', initial, (value) => { delete value.lifecycle.failedUpgrade.previousCapsuleId; }],
  ['missing predecessor root', initial, (value) => { delete value.lifecycle.failedUpgrade.previousSemanticRoot; }],
  ['initial migration', initial, (value) => { value.lifecycle.migration = upgrade.lifecycle.migration; }],
  ['missing migration', initial, (value) => { delete value.lifecycle.migration; }],
  ['disabled preservation', initial, (value) => { value.lifecycle.failedUpgrade.preservePrevious = false; }],
  ['missing upgrade rollback ID', upgrade, (value) => { value.lifecycle.failedUpgrade.previousCapsuleId = null; }],
  ['missing upgrade rollback root', upgrade, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = null; }],
  ['different upgrade rollback ID', upgrade, (value) => { value.lifecycle.failedUpgrade.previousCapsuleId = 'other'; }],
  ['different upgrade rollback root', upgrade, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = `sha256:${'8'.repeat(64)}`; }],
]) {
  const release = structuredClone(source);
  mutate(release);
  assert.equal(validateCapsuleReleaseContract(release).ok, false, name);
}

console.log('capsule-release-contract.test: ok');
