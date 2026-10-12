import assert from 'node:assert/strict';
import policy from '../fixtures/canonical-generation-policy.json' with { type: 'json' };
import { philox4x32, samplingWord, samplingUniform } from '../../src/inference/sampling-rng.js';
import { sampleCanonicalLogits, selectCanonicalCandidate } from '../../src/inference/token-sampling.js';
import { sampleCanonicalGenerationStep } from '../../src/inference/generation-step.js';

assert.deepEqual(philox4x32([0, 0, 0, 0], [0, 0]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
assert.deepEqual(philox4x32(Array(4).fill(0xffffffff), Array(2).fill(0xffffffff)),
  [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
for (const t of [0, 1, 0xffffffff, 0x100000000, Number.MAX_SAFE_INTEGER]) {
  for (let purpose = 0; purpose < 4; purpose++) {
    for (let draw = 0; draw < 12; draw++) {
      const expected = philox4x32([t >>> 0, Math.floor(t / 0x100000000), purpose, Math.floor(draw / 4)], [0x89abcdef, 0x01234567])[draw % 4];
      assert.equal(samplingWord('0123456789abcdef', t, purpose, draw), expected);
      assert.equal(samplingUniform('0123456789abcdef', t, purpose, draw), expected / 4294967296);
    }
  }
}
for (const seed of [0, '0', 'FFFFFFFFFFFFFFFF', undefined]) assert.throws(() => samplingWord(seed, 0, 0, 0));
for (const args of [[-1, 0, 0], [0, 4, 0], [0, 0, 0x400000000], [0.5, 0, 0]]) {
  assert.throws(() => samplingWord(policy.seedHex64, ...args));
}
const p = { ...policy, eosTokenIds: [], stopTokenIds: [], toolTokenIds: [] };
const sample = (scores, changes = {}, history = [], t = 0) => sampleCanonicalGenerationStep(scores, history, { ...p, ...changes }, t);
assert.equal(sample([2, 2, 1], { temperature: 0 }), 0);
assert.equal(sample([2, 2, 1], { temperature: 0, suppressTokenIds: [0] }), 1);
assert.equal(sample([2, 2, 1], { temperature: 0, padTokenId: 0 }), 1);
assert.equal(sample([2, 1], { temperature: Number.MIN_VALUE }), 0);
for (const scores of [[NaN, 3], [Infinity, 2], [-Infinity, -Infinity], [Number.MAX_VALUE, 1]]) assert.throws(() => sample(scores));
assert.throws(() => sample([1, 2], { suppressTokenIds: [0, 1] }));
assert.throws(() => sample([1, NaN], { suppressTokenIds: [1] }));
assert.throws(() => sample([1, 2], {}, [2]));
assert.throws(() => sampleCanonicalLogits([1, 2], [], p, 0, 2));
const scores = Float32Array.from([4, 3, 2]);
assert.equal(sample(scores, { temperature: 0, repetitionPenalty: 2 }, [0, 0, 0]), 1);
assert.equal(sample(scores, { temperature: 0, repetitionPenalty: 2, repetitionPenaltyWindow: 1 }, [0, 1]), 0);
assert.equal(sample(scores, { temperature: 0, presencePenalty: 2, presencePenaltyWindow: 1 }, [0, 1]), 0);
assert.equal(sample(scores, { temperature: 0, presencePenalty: 2, presencePenaltyWindow: 0 }, [0, 1]), 0);
assert.deepEqual([...scores], [4, 3, 2]);
assert.equal(sample([4, 3, 2, 1], { topK: 2, topP: 0.7 }), 0);
assert.equal(sample([4, 3, 2, 1], { topK: 1 }), 0);
const candidates = [{token: 0, logit: 0, prob: 0}, {token: 1, logit: 0, prob: 0.5}, {token: 2, logit: 0, prob: 0.5}];
assert.equal(selectCanonicalCandidate(candidates, 0), 1);
assert.equal(selectCanonicalCandidate(candidates, 0.5), 2);
assert.equal(selectCanonicalCandidate(candidates, 0xffffffff / 0x100000000), 2);
assert.throws(() => selectCanonicalCandidate(candidates, 1));
assert.throws(() => selectCanonicalCandidate([{ token: 0, logit: 0, prob: 0 }], 0));
const ordinary = Array.from({ length: 16 }, (_, t) => sample([2, 1, 0, -1], {}, [0, 1], t));
for (const k of [1, 2, 4]) {
  const interleaved = [];
  for (let t = 0; t < 16; t += k) {
    for (let j = t; j < Math.min(16, t + k); j++) {
      sampleCanonicalLogits([2, 1, 0, -1], [0, 1], p, j, 1);
      const selected = sample([2, 1, 0, -1], {}, [0, 1], j);
      assert.equal(sample([2, 1, 0, -1], {}, [0, 1], j), selected);
      interleaved.push(selected);
    }
  }
  assert.deepEqual(interleaved, ordinary);
}
console.log('canonical-sampling: Philox vectors, counters, penalties, endpoints, invalid distributions and draw scheduling passed');
