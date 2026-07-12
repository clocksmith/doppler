import assert from 'node:assert/strict';

import {
  compareVerifiedWgslRollouts,
  exactMcNemarPValue,
} from '../../tools/lib/wgsl-rollout-comparison.js';

function reward(passed) {
  return {
    reduction: { scalarReward: passed ? 1 : 0, blocked: false },
    components: [{
      id: 'exact_reference_match',
      normalizedValue: passed ? 1 : 0,
    }],
  };
}

function group(taskId, policyHash, referencePolicyHash, outcomes) {
  return {
    taskId,
    datasetHash: 'a'.repeat(64),
    policyHash,
    referencePolicyHash,
    sampling: { seed: 11, temperature: 0.8, topP: 0.95, maxTokens: 64 },
    samples: outcomes.map((passed, index) => ({
      sampleId: `${taskId}-${index}`,
      rewardVector: reward(passed),
    })),
  };
}

const referenceHash = 'b'.repeat(64);
const candidateHash = 'c'.repeat(64);
const reference = [
  group('task-1', referenceHash, referenceHash, [false, false]),
  group('task-2', referenceHash, referenceHash, [true, false]),
];
const candidate = [
  group('task-1', candidateHash, referenceHash, [true, false]),
  group('task-2', candidateHash, referenceHash, [true, true]),
];
const compared = compareVerifiedWgslRollouts(reference, candidate);
assert.equal(compared.reference.passAt1, 0.5);
assert.equal(compared.candidate.passAt1, 1);
assert.equal(compared.effects.passAt1, 0.5);
assert.equal(compared.paired.passAt1.referenceOnly, 0);
assert.equal(compared.paired.passAt1.candidateOnly, 1);
assert.equal(compared.paired.passAt1.exactMcNemarP, 1);
assert.equal(compared.reference.passAtK, 0.5);
assert.equal(compared.candidate.passAtK, 1);
assert.equal(compared.paired.samples.candidateOnly, 2);
assert.equal(exactMcNemarPValue(0, 6), 0.03125);
assert.equal(exactMcNemarPValue(0, 0), 1);

assert.throws(
  () => compareVerifiedWgslRollouts(reference, [candidate[0]]),
  /task counts differ/
);

console.log('wgsl-rollout-comparison.test: ok');
