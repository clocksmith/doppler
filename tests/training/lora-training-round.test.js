import assert from 'node:assert/strict';
import { createLoRATrainingRound, validateLoRATrainingRound, validateLoRAContributions } from '../../src/experimental/training/lora-training-round.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

const hash = (value) => computeCanonicalSha256(value);
const assignments = [
  { participantId: 'metal', sampleHash: hash('a'), supervisedTokenCount: 2 },
  { participantId: 'vulkan', sampleHash: hash('b'), supervisedTokenCount: 1 },
];
const round = createLoRATrainingRound({ jobId: 'community-fixture', roundId: 'step-0',
  baseManifestHash: hash('base'), trainingConfigHash: hash('training'),
  checkpoint: { stepCount: 0, adapter: { A: 'opaque fixture', B: 'opaque fixture' } }, assignments });
validateLoRATrainingRound(round);
const contributions = assignments.map((entry) => ({ ...entry,
  schema: 'doppler.lora-gradient-contribution/v1', roundHash: round.roundHash, loss: 1.5,
  gradients: { A: {}, B: {} } }));
assert.deepEqual(validateLoRAContributions(round, contributions.toReversed()), contributions);
assert.throws(() => validateLoRAContributions(round, contributions.slice(0, 1)), /incomplete/);
assert.throws(() => validateLoRAContributions(round, [contributions[0], contributions[0]]), /duplicate/);
for (const change of [
  { roundHash: hash('old') }, { participantId: 'unknown' }, { sampleHash: hash('other data') },
  { supervisedTokenCount: 8 }, { loss: NaN }, { schema: 'wrong' },
]) {
  assert.throws(() => validateLoRAContributions(round, [{ ...contributions[0], ...change }, contributions[1]]));
}
for (const mutate of [
  (r) => { r.checkpoint.stepCount++; },
  (r) => { r.baseManifestHash = hash('other weights'); },
  (r) => { r.trainingConfigHash = hash('other optimizer'); },
  (r) => { r.assignments.push(r.assignments[0]); },
  (r) => { r.assignments[0].supervisedTokenCount = 0; },
]) {
  const invalid = structuredClone(round); mutate(invalid);
  assert.throws(() => validateLoRATrainingRound(invalid));
}
assignments[0].participantId = 'changed after construction';
assert.equal(round.assignments[0].participantId, 'metal');
console.log('lora-training-round.test: ok (protocol only; no GPU claim)');
