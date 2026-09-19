import assert from 'node:assert/strict';
import { resetSequenceState } from '../../src/inference/pipelines/text/sequence-state.js';

const truncated = [];
const state = { isGenerating: false, currentSeqLen: 8, kvCache: { truncate: n => truncated.push(n) } };
for (const invalid of ['invalid', '2', NaN, Infinity, -Infinity, -1, 1.9, null, undefined, {}, Number.MAX_SAFE_INTEGER + 1]) {
  assert.throws(() => resetSequenceState(state, invalid), /non-negative safe integer/);
  assert.equal(state.currentSeqLen, 8);
  assert.deepEqual(truncated, []);
}
assert.throws(() => resetSequenceState(state, 9), /exceeds/);
state.isGenerating = true;
assert.throws(() => resetSequenceState(state, 0), /in progress/);
assert.deepEqual(truncated, []);
state.isGenerating = false;
resetSequenceState(state, 3);
resetSequenceState(state, 0);
assert.deepEqual(truncated, [3, 0]);
assert.equal(state.currentSeqLen, 0);
const failure = { isGenerating: false, currentSeqLen: 8, kvCache: { truncate() { throw new Error('cache failed'); } } };
assert.throws(() => resetSequenceState(failure, 2), /cache failed/);
assert.equal(failure.currentSeqLen, 8);
