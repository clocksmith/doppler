import assert from 'node:assert/strict';
import { validatePackReleaseContract } from '../../src/config/pack-release-contract.js';
import { createPackReleaseFixture } from '../helpers/pack-v2-fixture.js';

const upgrade = createPackReleaseFixture();
assert.equal(validatePackReleaseContract(upgrade).ok, true);
const initial = structuredClone(upgrade);
initial.lifecycle.supersedes = null;
initial.lifecycle.migration = null;
initial.lifecycle.failedUpgrade.previousPackId = null;
initial.lifecycle.failedUpgrade.previousSemanticRoot = null;
assert.deepEqual(validatePackReleaseContract(initial), { ok: true, errors: [] });

for (const [name, source, mutate] of [
  ['missing predecessor', initial, (value) => { delete value.lifecycle.supersedes; }],
  ['invented predecessor ID', initial, (value) => { value.lifecycle.failedUpgrade.previousPackId = 'invented'; }],
  ['invented predecessor root', initial, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = upgrade.lifecycle.supersedes.semanticRoot; }],
  ['missing predecessor ID', initial, (value) => { delete value.lifecycle.failedUpgrade.previousPackId; }],
  ['missing predecessor root', initial, (value) => { delete value.lifecycle.failedUpgrade.previousSemanticRoot; }],
  ['initial migration', initial, (value) => { value.lifecycle.migration = upgrade.lifecycle.migration; }],
  ['missing migration', initial, (value) => { delete value.lifecycle.migration; }],
  ['disabled preservation', initial, (value) => { value.lifecycle.failedUpgrade.preservePrevious = false; }],
  ['missing upgrade rollback ID', upgrade, (value) => { value.lifecycle.failedUpgrade.previousPackId = null; }],
  ['missing upgrade rollback root', upgrade, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = null; }],
  ['different upgrade rollback ID', upgrade, (value) => { value.lifecycle.failedUpgrade.previousPackId = 'other'; }],
  ['different upgrade rollback root', upgrade, (value) => { value.lifecycle.failedUpgrade.previousSemanticRoot = `sha256:${'8'.repeat(64)}`; }],
]) {
  const release = structuredClone(source);
  mutate(release);
  assert.equal(validatePackReleaseContract(release).ok, false, name);
}

console.log('pack-release-contract.test: ok');
