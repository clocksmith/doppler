// node --test tools/secret-shared/tests/   (plain CPU; opens no GPU device)
import assert from 'node:assert/strict';
import test from 'node:test';
import { acceptGreedy, argmaxRows, cloneState, padDraft, promptLookupDraft, speculativeDecode, summarize } from '../requester/speculate.js';

// A toy greedy model over a vocabulary of 11: the next token is a fixed function of the column's token. It records
// pass widths and checks the speculative/commit protocol: a speculative forward must be committed before the next one.
function toyModel(next, vocab = 11) {
  const widths = [];
  return {
    c: { vocab },
    widths,
    async forward(state, tokens, { keep, speculative }) {
      assert.equal(state.snapshot, null, 'previous pass not committed');
      assert.equal(keep, tokens.length);
      widths.push(tokens.length);
      state.snapshot = speculative ? { pos: state.pos, n: tokens.length } : null;
      state.pos += tokens.length;
      const logits = new Float32Array(tokens.length * vocab);
      tokens.forEach((t, r) => { logits[r * vocab + next(t)] = 1; });
      return logits;
    },
    commit(state, accepted) { state.pos = state.snapshot.pos + accepted; state.snapshot = null; }
  };
}

const sequential = (next, first, eos, max) => {
  const out = [first];
  while (!eos.includes(out.at(-1)) && out.length < max) out.push(next(out.at(-1)));
  return out;
};

test('prompt lookup, padding and greedy acceptance', () => {
  assert.deepEqual(promptLookupDraft([1, 2, 3, 4, 5, 9, 2, 3], 3), [4, 5, 9]);
  assert.deepEqual(promptLookupDraft([7, 8, 9], 2), []);
  assert.deepEqual(padDraft([4], 3, [1, 2]), { draft: [4, 4, 4], padded: 2 });
  assert.deepEqual(padDraft([], 2, [1, 2]), { draft: [2, 2], padded: 2 });
  assert.deepEqual(acceptGreedy([5, 6, 7], [5, 6, 9, 1]), { accepted: 2, tokens: [5, 6, 9] });
  assert.deepEqual(acceptGreedy([], [3]), { accepted: 0, tokens: [3] });
  assert.deepEqual(argmaxRows(new Float32Array([0, 2, 1, 5, 0, 0]), 2, 3), [1, 0]);
});

test('every k gives the sequential greedy tokens with fixed-width passes', async () => {
  const next = t => (t * 7 + 3) % 11, expected = sequential(next, 2, [10], 32);
  for (const k of [0, 1, 4, 8]) {
    const model = toyModel(next), state = { pos: 5, snapshot: null };
    const { tokens, passes } = await speculativeDecode(model, state, { promptIds: [2, 3, 5, 2, 3], first: 2, k, eos: [10], maxTokens: 32 });
    assert.deepEqual(tokens, expected);
    assert.deepEqual([...new Set(model.widths)], [k + 1]);
    assert.equal(passes.reduce((s, p) => s + p.emitted, 0), tokens.length - 1);
    assert.equal(state.pos, 5 + passes.reduce((s, p) => s + p.accepted + 1, 0));
  }
});

test('a right draft is accepted whole and the answer stops at the token limit', async () => {
  const model = toyModel(t => (t + 1) % 11, 11);
  const draft = (context, k) => Array.from({ length: k }, (_, i) => (context.at(-1) + 1 + i) % 11);
  const { tokens, passes } = await speculativeDecode(model, { pos: 1, snapshot: null }, { promptIds: [0], first: 1, k: 4, eos: [], maxTokens: 9, draft });
  assert.deepEqual(tokens, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(passes.map(p => [p.accepted, p.emitted]), [[4, 5], [4, 3]]);
  assert.deepEqual(summarize(tokens, passes).acceptedHistogram, [0, 0, 0, 0, 2]);
});

test('cloneState copies the KV and DeltaNet state', () => {
  const state = { pos: 3, conv: [new Float32Array([1, 2]), null], rec: [new Float32Array([3]), null],
    k: [null, { data: new Uint16Array([1, 2, 3, 0]), len: 3 }], v: [null, { data: new Uint16Array([4, 5, 6, 0]), len: 3 }], snapshot: null };
  const copy = cloneState(state);
  copy.conv[0][0] = 9; copy.k[1].data[0] = 9;
  assert.equal(state.conv[0][0], 1); assert.equal(state.k[1].data[0], 1);
  assert.equal(copy.k[1].len, 3); assert.equal(copy.pos, 3);
});
