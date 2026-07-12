import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { sha256BytesHex } from '../../src/utils/sha256.js';

const v9 = JSON.parse(await readFile('tools/policies/wgsl-repair-v9-policy.json', 'utf8'));
const v10 = JSON.parse(await readFile('tools/policies/wgsl-repair-v10-policy.json', 'utf8'));
const derivation = v10.methods.rollout.maxTokensDerivation;
const sourceBytes = new Uint8Array(await readFile(derivation.sourcePath));

assert.equal(v10.policyId, 'doppler-wgsl-repair-v10');
assert.equal(v10.status, 'frozen');
assert.equal(v10.models.primary.modelId, 'Qwen/Qwen3.5-9B');
assert.equal(v10.models.primary.revision, 'c202236235762e1c871ad0ccb60c8ee5ba337b9a');
assert.equal(v10.methods.sft.evaluationCheckpoint.parentPolicyId, v9.policyId);
assert.equal(
  v10.methods.sft.evaluationCheckpoint.policyHash,
  'fcba6cbbfee3b52de597be8df7400ae601f6fdbc15989946da543cb299eebc87'
);
assert.equal(v10.methods.rollout.maxTokens, 64);
assert.equal(derivation.maximumTargetTokensIncludingEos, 41);
assert.equal(derivation.marginTokens, 23);
assert.equal(
  derivation.maximumTargetTokensIncludingEos + derivation.marginTokens,
  v10.methods.rollout.maxTokens
);
assert.equal(derivation.holdoutOutcomesUsed, false);
assert.equal(derivation.sourceRows, 1200);
assert.equal(await sha256BytesHex(sourceBytes), derivation.sourceSha256);

const comparableV9 = structuredClone(v9);
const comparableV10 = structuredClone(v10);
comparableV10.policyId = comparableV9.policyId;
comparableV10.claimBoundary = comparableV9.claimBoundary;
delete comparableV10.methods.sft.evaluationCheckpoint;
delete comparableV10.methods.rollout.maxTokensDerivation;
comparableV10.methods.rollout.maxTokens = comparableV9.methods.rollout.maxTokens;
assert.deepEqual(
  comparableV10,
  comparableV9,
  'V10 may change only the declared evaluation checkpoint and rollout ceiling provenance.'
);

console.log('wgsl-repair-v10-policy.test: ok');
