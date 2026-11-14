import assert from 'node:assert/strict';
import { ExpertFrequencyIndex } from '../../src/loader/experts/expert-frequency-index.js';

const index = new ExpertFrequencyIndex(3);
for (const key of ['a', 'b', 'c']) { index.observe(key); index.admit(key); }
assert.equal(index.oldestUnpinned(() => false), 'a');
index.observe('a');
assert.equal(index.oldestUnpinned(() => false), 'b');
assert.equal(index.oldestUnpinned(key => key === 'b'), 'c');
assert.equal(index.oldestUnpinned(() => true), null);
assert.equal(index.residentSize, 3);
assert.throws(() => index.oldestUnpinned(key => { if (key === 'c') throw new Error('pin probe'); return true; }), /pin probe/);
assert.equal(index.residentSize, 3);
assert.equal(index.oldestUnpinned(() => false), 'b');
assert.equal(index.remove('a'), true);
assert.equal(index.remove('a'), false);
assert.equal(index.observe('a'), 3);
index.admit('a'); index.admit('a');
assert.equal(index.historySize, 3); assert.equal(index.residentSize, 3);
assert.throws(() => index.observe('d'), /declared key limit/);
assert.throws(() => index.admit('unknown'), /observed/);
assert.throws(() => index.observe(''), /nonempty/);
for (const limit of [0, -1, 1.5, Infinity]) assert.throws(() => new ExpertFrequencyIndex(limit), /positive/);
index.clear();
assert.equal(index.historySize, 0); assert.equal(index.residentSize, 0);
assert.equal(index.oldestUnpinned(() => false), null);
assert.equal(index.observe('d'), 1);

// Deterministic mixed operations against a simple independent scan oracle.
const heap = new ExpertFrequencyIndex(257), history = new Map(), resident = new Set();
let sequence = 0, seed = 17;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
for (let step = 0; step < 10000; step++) {
  const key = String(random() % 257), operation = random() % 3;
  if (operation === 0 || !history.has(key)) {
    const prior = history.get(key);
    history.set(key, { frequency: (prior?.frequency ?? 0) + 1, sequence: ++sequence });
    assert.equal(heap.observe(key), history.get(key).frequency);
  } else if (operation === 1) { resident.add(key); heap.admit(key); }
  else { assert.equal(heap.remove(key), resident.delete(key)); }
  const pinned = key => Number(key) % 7 === step % 7;
  const expected = [...resident].filter(key => !pinned(key)).sort((a, b) =>
    history.get(a).frequency - history.get(b).frequency || history.get(a).sequence - history.get(b).sequence)[0] ?? null;
  assert.equal(heap.oldestUnpinned(pinned), expected);
  assert.equal(heap.historySize, history.size);
  assert.equal(heap.residentSize, resident.size);
}
console.log('Expert frequency indexed heap: priority, retained history, pins, failure restoration and 10,000 mixed operations passed.');
