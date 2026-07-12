import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  buildWgslRolloutSampling,
  resolveWgslRolloutTaskPath,
} from '../../tools/run-wgsl-repair-experiment.js';

const corpusRoot = 'reports/training/wgsl-repair/doppler-wgsl-repair-v9/corpus-v1';
const v10 = JSON.parse(await readFile('tools/policies/wgsl-repair-v10-policy.json', 'utf8'));
const v11 = JSON.parse(await readFile('tools/policies/wgsl-repair-v11-policy.json', 'utf8'));

function parseJsonl(text) {
  return text.trim().split('\n').map((line) => JSON.parse(line));
}

function familyIds(rows) {
  return new Set(rows.map((row) => row.kernelFamilyId));
}

function overlap(left, right) {
  return [...left].filter((value) => right.has(value));
}

assert.equal(v11.policyId, 'doppler-wgsl-repair-v11');
assert.equal(v11.status, 'frozen');
assert.equal(v11.methods.rollout.taskSet, 'diagnostic');
assert.equal(v11.corpus.splits.diagnostic.visibleToTraining, true);
assert.equal(v11.corpus.splits['public-test'].visibleToTraining, false);
assert.equal(
  resolveWgslRolloutTaskPath(corpusRoot, v11.methods.rollout),
  `${corpusRoot}/diagnostic.jsonl`
);
assert.equal(
  resolveWgslRolloutTaskPath(corpusRoot, v10.methods.rollout),
  `${corpusRoot}/public-test.jsonl`
);
assert.deepEqual(
  buildWgslRolloutSampling(v11),
  buildWgslRolloutSampling(v10),
  'Task routing must not change the frozen sampling contract.'
);
assert.throws(
  () => resolveWgslRolloutTaskPath(corpusRoot, { taskSet: 'train' }),
  /must be diagnostic or public-test/
);

const [train, diagnostic, publicTest] = await Promise.all([
  readFile(`${corpusRoot}/train.jsonl`, 'utf8').then(parseJsonl),
  readFile(`${corpusRoot}/diagnostic.jsonl`, 'utf8').then(parseJsonl),
  readFile(`${corpusRoot}/public-test.jsonl`, 'utf8').then(parseJsonl),
]);
assert.equal(diagnostic.length, 285);
assert.equal(publicTest.length, 299);
assert.deepEqual(overlap(familyIds(train), familyIds(diagnostic)), []);
assert.deepEqual(overlap(familyIds(train), familyIds(publicTest)), []);
assert.deepEqual(overlap(familyIds(diagnostic), familyIds(publicTest)), []);

const comparableV10 = structuredClone(v10);
const comparableV11 = structuredClone(v11);
comparableV11.policyId = comparableV10.policyId;
comparableV11.claimBoundary = comparableV10.claimBoundary;
comparableV11.corpus.splits.diagnostic.visibleToTraining = false;
delete comparableV11.methods.rollout.taskSet;
comparableV11.verifier.roles = comparableV10.verifier.roles;
assert.deepEqual(
  comparableV11,
  comparableV10,
  'V11 may change only diagnostic-task routing and its evidence boundary.'
);

console.log('wgsl-repair-v11-policy.test: ok');
