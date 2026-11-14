import test from 'node:test';
import assert from 'node:assert/strict';
import { candidateSpread, diversity, perplexity, surprise } from '../public/viz/signals.js';

const token = (tokenId, probability, candidates = null) => ({ tokenId, probability, candidates });

test('surprise is −ln p and absent without a probability', () => {
  assert.equal(surprise(token(1, 1)), -0);
  assert.ok(Math.abs(surprise(token(1, 0.5)) - Math.LN2) < 1e-12);
  assert.equal(surprise(token(1, null)), null);
  assert.equal(surprise(token(1, 0)), null);
});

test('candidate spread renormalizes the listed subset and reports its coverage', () => {
  const spread = candidateSpread(token(1, 0.3, [{ probability: 0.3 }, { probability: 0.3 }]));
  assert.ok(Math.abs(spread.entropy - Math.LN2) < 1e-12, 'two equal candidates: ln 2 whatever they cover');
  assert.ok(Math.abs(spread.coverage - 0.6) < 1e-12);
  assert.equal(spread.count, 2);
  assert.equal(candidateSpread(token(1, 0.9, [{ probability: 0.9 }])), null, 'one candidate has no spread');
  assert.equal(candidateSpread(token(1, 0.9)), null);
});

test('perplexity uses only recorded tokens and states their range', () => {
  const result = perplexity([token(1, null), token(2, 0.5), token(3, 0.5), token(4, null)]);
  assert.ok(Math.abs(result.value - 2) < 1e-12);
  assert.deepEqual([result.tokens, result.first, result.last], [2, 1, 2]);
  assert.equal(perplexity([token(1, null)]), null);
});

test('diversity looks back over a bounded window', () => {
  const tokens = [1, 2, 1, 3].map((id) => token(id, 0.5));
  assert.deepEqual(diversity(tokens, 2), { distinct: 2 / 3, repeated: true, window: 3 });
  assert.deepEqual(diversity(tokens, 3, 2), { distinct: 1, repeated: false, window: 2 });
});

test('confusion is the recorded full-distribution entropy, absent otherwise', async () => {
  const { confusion } = await import('../public/viz/signals.js');
  assert.equal(confusion({ entropy: 1.5 }), 1.5);
  assert.equal(confusion({ entropy: null }), null);
  assert.equal(confusion({}), null);
});
