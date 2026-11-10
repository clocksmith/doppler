import test from 'node:test';
import assert from 'node:assert/strict';
import { dyadic, add, multiply, roundF32, fma, inverseRootInterval, inverseRootCandidates, compare } from './dyadic-diagnostic.js';

test('exact FMA preserves cancellation lost by separate F32 operations', () => {
  const a = 1 + 2 ** -23; const b = 1 - 2 ** -23;
  assert.equal(fma(a, b, -1), -(2 ** -46));
  assert.equal(Math.fround(Math.fround(a * b) - 1), 0);
});

test('F32 rounding includes ties, underflow and negative numbers', () => {
  for (const value of [1 + 2 ** -24, 1 + 3 * 2 ** -24, -(1 + 2 ** -24), 2 ** -150, 3 * 2 ** -150, 65504]) {
    assert.equal(roundF32(dyadic(value)), Math.fround(value));
  }
  assert.equal(roundF32(add(dyadic(2 ** 80), dyadic(-(2 ** 80)))), 0);
});

test('inverse root enclosure is proved by exact squared inequalities', () => {
  for (const value of [Math.fround(1e-6), 0.125, 0.7, 1, 2, 17, 1024]) {
    const { lower, upper } = inverseRootInterval(value);
    const one = dyadic(1); const x = dyadic(value);
    assert.ok(compare(multiply(multiply(lower, lower), x), one) <= 0n);
    assert.ok(compare(multiply(multiply(upper, upper), x), one) >= 0n);
    assert.ok(inverseRootCandidates(value).length >= 3);
  }
});
