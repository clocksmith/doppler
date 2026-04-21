import assert from 'node:assert/strict';

const { createHasher, hash } = await import('../../src/storage/blake3.js');

const bytes = new Uint8Array([1, 2, 3, 4]);
const hasher = createHasher();
hasher.update(bytes.subarray(0, 2));
hasher.update(bytes.subarray(2));

const first = hasher.finalize();
const second = hasher.finalize();
const expected = await hash(bytes);

assert.deepEqual(first, expected);
assert.deepEqual(second, expected);
assert.notEqual(first, second, 'finalize should return a defensive copy');
assert.throws(
  () => hasher.update(new Uint8Array([5])),
  /update called after finalize/
);

const emptyHasher = createHasher();
assert.deepEqual(emptyHasher.finalize(), await hash(new Uint8Array(0)));
assert.deepEqual(emptyHasher.finalize(), await hash(new Uint8Array(0)));

console.log('blake3.test: ok');
