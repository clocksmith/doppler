import assert from 'node:assert/strict';

const { toUniformArrayBuffer } = await import('../../src/gpu/uniform-cache.js');

const words = new Uint32Array([1, 2, 3]);
assert.equal(toUniformArrayBuffer(words), words.buffer);

const bytes = new Uint8Array(words.buffer);
const backing = new Uint8Array(bytes.byteLength + 1);
backing.set(bytes, 1);
const ranged = backing.subarray(1);
const arrayBuffer = toUniformArrayBuffer(ranged);
assert.notEqual(arrayBuffer, backing.buffer);
assert.deepEqual(Array.from(new Uint8Array(arrayBuffer)), Array.from(bytes));

console.log('uniform-array-buffer.test: ok');
