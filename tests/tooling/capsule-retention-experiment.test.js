import assert from 'node:assert/strict';
import { generateRetentionCandidates, validateExperimentSplit } from '../../tools/run-capsule-retention-experiment.js';

const digest = value => `sha256:${value.repeat(64)}`;
const scope = { runtimeHash: digest('a'), capsuleHash: digest('b'), targetPlanDigest: digest('c') };
const candidates = generateRetentionCandidates({ ...scope, retentionBytes: [null, 134217728] });
assert.equal(candidates.length, 2);
for (const candidate of candidates) {
  assert.equal(candidate.runtimeHash, scope.runtimeHash);
  assert.equal(candidate.capsuleHash, scope.capsuleHash);
  assert.equal(candidate.targetPlanDigest, scope.targetPlanDigest);
}
for (const retentionBytes of [[1], [null], [null, null], [null, -1], [null, NaN], [null, '128']]) {
  assert.throws(() => generateRetentionCandidates({ ...scope, retentionBytes }));
}
const row = (id, query) => ({ id, reference: { input: { query, documents: ['fixed document'] },
  source: { files: [{ path: 'weights', hash: digest('d') }] }, scoringConfig: { score: 'true_logit' },
  tolerances: { scoreMaxAbs: 1 } } });
const tuning = [row('tune', 'tuning input')], heldout = [row('heldout', 'unseen input')];
validateExperimentSplit(tuning, heldout);
assert.throws(() => validateExperimentSplit(tuning, [row('heldout', 'tuning input')]), /disjoint/);
assert.throws(() => validateExperimentSplit(tuning, [row('tune', 'different input')]), /unique/);
assert.throws(() => validateExperimentSplit([], heldout), /Both/);
for (const change of [
  value => { value.reference.tolerances.scoreMaxAbs = 2; },
  value => { value.reference.source.files[0].hash = digest('e'); },
  value => { value.reference.scoringConfig.score = 'logit_difference'; },
]) {
  const changed = structuredClone(heldout); change(changed[0]);
  assert.throws(() => validateExperimentSplit(tuning, changed), /unchanged|must match/);
}
console.log('capsule-retention-experiment.test: ok (split and identity gates; no hardware measurements)');
