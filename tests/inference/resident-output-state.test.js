import assert from 'node:assert/strict';
import { createBundledIncrementalDecoder } from '../../src/inference/tokenizers/bundled/incremental-decoder.js';
import { createResidentOutputState } from '../../src/inference/resident-output-state.js';
import fixture from '../fixtures/canonical-generation-policy.json' with { type: 'json' };

const table = ['hello', 'ST', 'OPafter', 'STO', 'X', '<0xE2>', '<0x82>', '<0xAC>', '<0xEF>', '<0xBB>', '<0xBF>'];
const decoder = () => createBundledIncrementalDecoder({ tokenForId: id => table[id], byteLevel: false,
  byteDecoder: new Map(), wordPiece: false, splitEveryCharacter: false, wordPiecePrefix: '##' });
const policy = { ...fixture, eosTokenIds: [], stopTokenIds: [], toolTokenIds: [] };
const create = changes => createResidentOutputState(decoder(), decoder(), { ...policy, ...changes }, table.length);
const parent = create();
assert.deepEqual(parent.accept(1), { delta: '', textStop: false });
const digest = parent.digest(), child = parent.clone();
assert.deepEqual(child.accept(2), { delta: '', textStop: true });
assert.equal(child.finalize('stop-sequence'), '');
assert.equal(parent.digest(), digest);
assert.deepEqual(parent.accept(4), { delta: 'STX', textStop: false });
assert.equal(parent.finalize('max-tokens'), '');
const flush = create(); assert.equal(flush.accept(3).delta, '');
assert.equal(flush.finalize('max-tokens'), 'STO');
// Full match inside a single chunk hides both the match and following text.
table.push('beforeSTOPafter');
const inside = create(); assert.deepEqual(inside.accept(11), {delta: 'before', textStop: true});
const earliest = create({ stopSequences: ['beforeSTOP', 'before'] });
assert.deepEqual(earliest.accept(11), {delta: '', textStop: true});
// Partial UTF-8 and BOM control are copied without replaying accepted IDs.
for (const prefix of [[], [0], [8, 9, 10]]) {
  const a = decoder(); prefix.forEach(id => a.push(id));
  a.push(5); a.push(6);
  const b = a.clone(); assert.deepEqual(a.snapshot(), b.snapshot());
  assert.equal(a.push(7), '€'); assert.equal(b.push(7), '€');
  assert.deepEqual(a.snapshot(), b.snapshot());
  assert.equal(a.finish(), ''); assert.equal(b.finish(), '');
}
const bounded = create({ maxVisibleOutputUtf16: 2 });
assert.throws(() => bounded.accept(0), /Visible UTF-16/);
const rawBounded = create({ maxRawOutputUtf16: 2 });
assert.throws(() => rawBounded.accept(0), /Raw UTF-16/);
console.log('resident-output-state: clone isolation, partial UTF-8, stop withholding and hard output guards passed');
