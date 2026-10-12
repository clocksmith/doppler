import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createResidentTargetState } from '../../src/inference/resident-target-state.js';
import { createResidentOutputState } from '../../src/inference/resident-output-state.js';
import { createBundledIncrementalDecoder } from '../../src/inference/tokenizers/bundled/incremental-decoder.js';

const base = JSON.parse(readFileSync(new URL('../fixtures/canonical-generation-policy.json', import.meta.url)));
const generationId = `sha256:${'c'.repeat(64)}`;
const policy = { ...base, stopSequences: [], eosTokenIds: [], stopTokenIds: [], toolTokenIds: [] };
const decoder = () => createBundledIncrementalDecoder({ tokenForId: id => String.fromCharCode(65 + id),
  byteLevel: false, byteDecoder: new Map(), wordPiece: false, splitEveryCharacter: false, wordPiecePrefix: '##' });
const state = (p = policy) => createResidentTargetState([1, 2], p, 100, generationId,
  createResidentOutputState(decoder(), decoder(), p, 100));
const logits = () => Float32Array.from({ length: 100 }, (_, id) => Math.sin(id) * 3);
const force = id => { const row = new Float32Array(100).fill(-Infinity); row[id] = 0; return row; };
for (const k of [1, 2, 4]) {
  for (let reject = 0; reject <= k; reject++) {
    const parent = state();
    const first = parent.bootstrap(logits());
    const sequential = parent.clone(), scratch = parent.clone();
    const expected = [];
    for (let j = 0; j <= k; j++) expected.push(sequential.advance(sequential.getCursor().cursorId, logits()));
    const proposal = expected.slice(0, k).map(item => item.acceptedTokenIds[0]);
    if (reject < k) proposal[reject] = (proposal[reject] + 1) % 100;
    const result = scratch.verifyBlock(first.successor.cursorId, proposal, Array.from({ length: k + 1 }, logits));
    const count = reject < k ? reject + 1 : k + 1;
    assert.deepEqual(result.acceptedTokenIds, expected.slice(0, count).flatMap(item => item.acceptedTokenIds));
    assert.equal(result.delta, expected.slice(0, count).map(item => item.delta).join(''));
    assert.equal(result.outputDigest, expected[count - 1].outputDigest);
    assert.equal(result.successor.acceptedTokenDigest, expected[count - 1].successor.acceptedTokenDigest);
    assert.equal(result.successor.revision, first.successor.revision + 1, 'one revision per block');
    assert.deepEqual(parent.getCursor(), first.successor, 'scratch never mutates its parent');
    assert.deepEqual(parent.getHistory(), [1, 2]);
    assert.deepEqual(scratch.getHistory(), [1, 2, first.acceptedTokenIds[0], ...result.acceptedTokenIds.slice(0, -1)]);
  }
}
for (let stop = 0; stop <= 4; stop++) {
  const p = { ...policy, eosTokenIds: [99] }, parent = state(p);
  parent.bootstrap(force(3));
  const scratch = parent.clone(), proposal = [4, 5, 6, 7];
  const rows = [...proposal, 8].map(force); rows[stop] = force(99);
  const result = scratch.verifyBlock(parent.getCursor().cursorId, proposal, rows);
  assert.equal(result.acceptedTokenIds.length, stop + 1);
  assert.equal(result.successor.pendingTokenId, 99);
  assert.equal(result.successor.stopReason, 'eos-token');
  assert.equal(scratch.getHistory().includes(99), false, 'terminal token remains pending');
}
{
  const value = state({ ...policy, materializedInputCapacity: 2 });
  const result = value.bootstrap(force(3));
  assert.equal(result.successor.stopReason, 'context-limit');
  assert.throws(() => value.advance(result.successor.cursorId, logits()), /Terminal/);
}
{
  const value = state({ ...policy, maxRawOutputUtf16: 0 });
  assert.throws(() => value.bootstrap(force(3)), /Raw UTF-16/);
  assert.throws(() => value.getCursor(), /interrupted/);
  assert.throws(() => value.clone(), /interrupted/);
}
{
  const value = state();
  const first = value.bootstrap(force(3));
  assert.throws(() => value.advance(generationId, logits()), /cursor mismatch/);
  assert.deepEqual(value.getCursor(), first.successor, 'stale request cannot mutate decoder');
  const row = logits(); row[0] = NaN;
  assert.throws(() => value.advance(first.successor.cursorId, row), /Invalid post-model/);
  assert.throws(() => value.getHistory(), /interrupted/);
}
console.log('resident-target-state: ordinary/block sampling parity, isolated decoder branches, pending terminals and interruption passed');
