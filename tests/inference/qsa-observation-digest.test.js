import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { hashQsaObservationBytes, observeLearnedQsa } from '../helpers/learned-qsa-observation.js';

// Index 256 must not collide with index 0 by truncating each value to one byte.
assert.notEqual(hashQsaObservationBytes(new Uint32Array([0])), hashQsaObservationBytes(new Uint32Array([256])));
const storage = new Uint32Array([99, 256, 0x01020304, 88]);
const selected = storage.subarray(1, 3);
const expectedBytes = new ArrayBuffer(8);
const expected = new DataView(expectedBytes);
expected.setUint32(0, 256, true);
expected.setUint32(4, 0x01020304, true);
const digest = createHash('sha256').update(new Uint8Array(expectedBytes)).digest('hex');
assert.equal(hashQsaObservationBytes(selected), digest);
assert.equal(hashQsaObservationBytes(expectedBytes), digest);
assert.equal(hashQsaObservationBytes(new DataView(storage.buffer, 4, 8)), digest);
for (const position of [-1, 1.5, NaN]) {
  assert.throws(() => observeLearnedQsa([], {}, '', [position]), /nonnegative integers/);
}
console.log('QSA observation hashes retain high bits and respect view byte offsets.');
