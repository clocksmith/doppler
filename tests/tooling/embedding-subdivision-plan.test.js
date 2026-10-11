import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateEmbeddingSubdivisionPlan, persistEmbeddingQualificationCheckpoint } from '../../tools/qualify-embedding-browser.js';

const plan = [[2, 0], [1, 3]];
assert.deepEqual(validateEmbeddingSubdivisionPlan(plan, 4), plan);
const detached = validateEmbeddingSubdivisionPlan(plan, 4);
detached[0][0] = 9;
assert.equal(plan[0][0], 2);
for (const [invalid, pattern] of [
  [null, /two units/], [[[0, 1, 2, 3]], /two units/],
  [[[0, 1], []], /original positions/], [[[0, 1], [2]], /every original/],
  [[[0, 1], [1, 3]], /unique/], [[[0, 1], [2, 4]], /in range/],
  [[[0, 1], [2, 3.1]], /in range/],
]) assert.throws(() => validateEmbeddingSubdivisionPlan(invalid, 4), pattern);
const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'embedding-checkpoint-test-'));
try {
  const first = { sequence: 1, digest: `sha256:${'1'.repeat(64)}` };
  const second = { sequence: 2, digest: `sha256:${'2'.repeat(64)}` };
  const filename = path.join(directory, 'release-checkpoint.json');
  assert.deepEqual(await persistEmbeddingQualificationCheckpoint(directory, null, first), first);
  await persistEmbeddingQualificationCheckpoint(directory, first, second);
  assert.deepEqual(JSON.parse(await fs.readFile(filename)), second);
  await assert.rejects(persistEmbeddingQualificationCheckpoint(directory, second, first), /rollback/);
  await assert.rejects(persistEmbeddingQualificationCheckpoint(directory, second, { ...second, digest: first.digest }), /fork/);
  assert.deepEqual(JSON.parse(await fs.readFile(filename)), second, 'rejections preserve the committed checkpoint');
  await assert.rejects(persistEmbeddingQualificationCheckpoint(directory, second, { sequence: 3, digest: 'bad' }), /verified/);
  await assert.rejects(persistEmbeddingQualificationCheckpoint(path.join(directory, 'absent'), second,
    { ...second, sequence: 3 }), /ENOENT/);
} finally { await fs.rm(directory, { recursive: true, force: true }); }
console.log('embedding-subdivision-plan.test: passed');
