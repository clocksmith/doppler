import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateRecordingBytes, redactedSummary, retain, retainedLimit } from '../public/run-files.js';

test('a rejected replacement preserves the previous recording and its budget', () => {
  const sizes = new Map([['other', retainedLimit - 64], ['old', 24]]);
  assert.throws(() => retain(sizes, 'new', 'x'.repeat(64), 'old'), /budget/);
  assert.deepEqual([...sizes], [['other', retainedLimit - 64], ['old', 24]]);
  retain(sizes, 'new', 2, 'old');
  assert.deepEqual([...sizes], [['other', retainedLimit - 64], ['new', 16]]);
});

test('count and nesting limits hold even for tiny recordings', () => {
  const sizes = new Map();
  for (let i = 0; i < 64; i++) retain(sizes, i, {});
  assert.throws(() => retain(sizes, 'new', {}), /64 recordings/);
  let deep = null;
  for (let i = 0; i < 130; i++) deep = [deep];
  assert.throws(() => estimateRecordingBytes(deep), /nesting/);
});

test('redacted exports contain only aggregated numeric values', () => {
  assert.deepEqual(redactedSummary({
    prompt: 'private', identity: 'private', tokens: [{ text: 'private', tokenId: 42 }],
    timing: { wallMs: 3, timeToFirstTokenMs: 'private', decodeMs: 2, extra: 'private' },
  }), { schema: 'doppler.observatory.redacted-summary/v1', answerTokens: 1, wallMs: 3, firstTokenMs: null, decodeMs: 2 });
});
