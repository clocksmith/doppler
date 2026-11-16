import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createVerifiedPieceReader } from '../../src/storage/verified-piece-reader.js';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const pieces = [new Uint8Array([1, 2, 3, 4]), new Uint8Array([5, 6, 7, 8])];
const manifestBytes = encode({ modelId: 'read-into', hashAlgorithm: 'sha256',
  shards: [{ filename: 'weights.bin', size: 8, hash: 'a'.repeat(64) }],
  metadata: { sourceRuntime: { mode: 'direct-source', sourceKind: 'gguf',
    sourceFiles: [{ index: 0, path: '/approved/model.gguf', filename: 'weights.bin',
      size: 8, hash: 'a'.repeat(64), hashAlgorithm: 'sha256' }] } } });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes),
  maxPieceBytes: 4, files: [{ path: 'weights.bin', size: 8,
    pieces: pieces.map((bytes, i) => ({ offset: i * 4, size: 4, identity: sha(bytes) })) }] });
const fixture = acquire => ({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes), acquire });
const reads = [];
const source = await createVerifiedPieceStorage(fixture(async piece => {
  reads.push(piece.offset);
  return pieces[piece.offset / 4];
}));
const target = new Uint8Array(9).fill(99);
assert.equal(await source.storage.loadShardRangeInto(0, 2, target.subarray(2, 7)), 5);
assert.deepEqual([...target], [99, 99, 3, 4, 5, 6, 7, 99, 99]);
assert.deepEqual(reads, [0, 4]);
assert.equal(source.getReceipt().peakReadBytes, 0, 'Caller storage needs no reader-owned range output');
assert.equal(source.getReceipt().requestedBytes, 5);
assert.equal(source.getReceipt().verifiedBytes, 8, 'Both complete pieces are authenticated');
assert.equal(await source.storage.loadShardRangeInto(0, 8, new Uint8Array()), 0);
await assert.rejects(source.storage.loadShardRangeInto(0, 7, new Uint8Array(2)), /exceeds/);
await source.storage.close();
await assert.rejects(source.storage.loadShardRangeInto(0, 0, new Uint8Array(1)), /closed/);

// Failure never publishes unverified bytes; the verified prefix remains explicit.
let corrupt = true;
const reader = await createVerifiedPieceReader(fixture(async piece =>
  corrupt && piece.offset === 4 ? new Uint8Array(4) : pieces[piece.offset / 4]));
const retry = new Uint8Array(8).fill(99);
await assert.rejects(reader.readInto('weights.bin', 0, retry), /integrity/);
assert.deepEqual([...retry], [1, 2, 3, 4, 99, 99, 99, 99]);
corrupt = false;
await reader.readInto('weights.bin', 0, retry);
assert.deepEqual([...retry], [1, 2, 3, 4, 5, 6, 7, 8]);
await assert.rejects(reader.readInto('weights.bin', 0, new Uint16Array(2)), /byte destination/);
await assert.rejects(reader.readInto('weights.bin', -1, retry), /outside/);
await reader.close();

// Close settles a borrowed destination before its caller can reuse it, even if
// the host ignores abort and supplies its payload later.
let started, release;
const acquiring = new Promise(resolve => { started = resolve; });
const host = new Promise(resolve => { release = resolve; });
const closing = await createVerifiedPieceReader(fixture(async () => { started(); return host; }));
const borrowed = new Uint8Array(4).fill(77);
let settled = false;
const pending = closing.readInto('weights.bin', 0, borrowed).then(
  () => { throw new Error('Cancelled acquisition completed'); },
  error => { settled = true; return error; });
await acquiring;
await closing.close();
assert.equal(settled, true);
assert.match((await pending).message, /closed/);
borrowed.fill(66);
release(pieces[0]);
await new Promise(resolve => setImmediate(resolve));
assert.deepEqual([...borrowed], [66, 66, 66, 66], 'No late writes after close');

// Cancellation after hashing also leaves caller storage untouched.
const abort = new AbortController();
const hashing = await createVerifiedPieceReader({ ...fixture(async () => pieces[0]), signal: abort.signal });
const originalDigest = crypto.subtle.digest;
crypto.subtle.digest = async function (...args) {
  const result = await originalDigest.apply(this, args);
  abort.abort(new Error('Cancelled before destination publication'));
  return result;
};
const unpublished = new Uint8Array(4).fill(55);
try { await assert.rejects(hashing.readInto('weights.bin', 0, unpublished), /Cancelled/); }
finally { crypto.subtle.digest = originalDigest; await hashing.close(); }
assert.deepEqual([...unpublished], [55, 55, 55, 55]);
console.log('verified-piece-read-into: authenticated destination, corruption, retry and cancellation passed');
