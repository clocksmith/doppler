import test from 'node:test';
import assert from 'node:assert/strict';
import { dotEnvelope, nearestHalf, halfValue, cacheDecision } from './numerical-discontinuity-core.js';

test('exact dyadic oracle detects cancellation and independently bounds both orders', () => {
  const result = dotEnvelope([2 ** 24, 1, -(2 ** 24)], [1, 1, 1], 0);
  assert.equal(result.exact, 1);
  assert.equal(result.forward, 0);
  assert.equal(result.reverse, 1);
  assert.equal(result.within, true);
  assert.equal(result.forwardWithin, true);
  assert.equal(result.reverseWithin, true);
  assert.equal(result.contains(1000), false);
});

test('independent F16 conversion round-trips every finite bit pattern including signed zero', () => {
  for (let word = 0; word <= 65535; word++) {
    if (((word >>> 10) & 31) !== 31) assert.equal(nearestHalf(halfValue(word)), word);
  }
});

test('all positive and negative adjacent-half midpoints use ties-to-even', () => {
  for (let lo = 0; lo < 31743; lo++) {
    const midpoint = (halfValue(lo) + halfValue(lo + 1)) / 2;
    const expected = lo % 2 ? lo + 1 : lo;
    assert.equal(nearestHalf(midpoint), expected);
    assert.equal(nearestHalf(-midpoint), expected | 32768);
  }
});

test('correct rounding of corrupted input cannot establish upstream correctness', () => {
  const original = 1;
  const corrupted = 1.01;
  const result = cacheDecision(corrupted, nearestHalf(corrupted), original, nearestHalf(original));
  assert.equal(result.ownCorrect, true);
  assert.equal(result.different, true);
  assert.equal(result.upstreamStatus, 'unresolved-no-independent-input-bound');
  assert.equal(cacheDecision(original, nearestHalf(original) ^ 1, original, nearestHalf(original)).ownCorrect, false);
});

test('unsupported arithmetic inputs fail explicitly', () => {
  assert.throws(() => dotEnvelope([1], [], 1), /shape/);
  assert.throws(() => dotEnvelope([NaN], [1], 1), /finite/);
  assert.throws(() => dotEnvelope([2 ** -149], [1], 0), /Subnormal/);
  assert.throws(() => nearestHalf(Infinity), /nonfinite/);
  assert.throws(() => nearestHalf(65520), /overflow/);
});

test('fixed independent synthetic cases exercise cancellation and varying magnitudes', () => {
  let state = 0x19b47a31;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 2 ** 32; };
  for (const size of [1, 7, 33, 257]) for (let sample = 0; sample < 32; sample++) {
    const input = Array.from({ length: size }, () => Math.fround((random() - 0.5) * 2 ** Math.floor(random() * 32 - 16)));
    const weights = Array.from({ length: size }, () => halfValue(Math.floor(random() * 31744) | (random() < 0.5 ? 32768 : 0)));
    const result = dotEnvelope(input, weights, 0);
    assert.ok(result.forwardWithin);
    assert.ok(result.reverseWithin);
  }
});
