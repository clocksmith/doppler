import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const data = [new Uint8Array([1,2,3,4]), new Uint8Array([5,6,7,8])];
const manifestBytes = encode({modelId:'test', hashAlgorithm:'sha256',shards:[{filename:'weights.bin',size:8,hash:'a'.repeat(64)}]});
const index = {schema:'doppler.verified-pieces/v1',manifestIdentity:sha(manifestBytes),maxPieceBytes:4,
 files:[{path:'weights.bin',size:8,pieces:data.map((bytes,i)=>({offset:i*4,size:4,identity:sha(bytes)}))}]};
const indexBytes=encode(index),indexIdentity=sha(indexBytes),reads=[];
const opened=await createVerifiedPieceStorage({manifestBytes,indexBytes,indexIdentity,acquire:async p=>{reads.push(p.offset);return data[p.offset/4];}});
await opened.storage.preflight();
assert.deepEqual(reads, [], 'preflight must not acquire unassigned shard tails');
assert.deepEqual([...new Uint8Array(await opened.storage.loadShardRange(0,1,2))],[2,3]);
assert.deepEqual(reads,[0]); // Hash checking does not fetch the unrelated second piece.
assert.equal(opened.getReceipt().verifiedBytes,4);
assert.equal(opened.getReceipt().activeReadBytes,0);
assert.equal(opened.getReceipt().peakReadBytes,2);
await opened.storage.close();await assert.rejects(opened.storage.loadShardRange(0,0,1),/closed|outside/);
await assert.rejects(createVerifiedPieceStorage({manifestBytes,indexBytes,indexIdentity:'sha256:'+'0'.repeat(64),acquire:async()=>data[0]}),/identity/);
const corrupt=await createVerifiedPieceStorage({manifestBytes,indexBytes,indexIdentity,acquire:async()=>new Uint8Array(4)});
await assert.rejects(corrupt.storage.loadShardRange(0,0,1),/integrity/);
assert.equal(corrupt.getReceipt().activeReadBytes,0, 'Failed verification releases read ownership');
const changed=encode({...index,manifestIdentity:'sha256:'+'0'.repeat(64)});
await assert.rejects(createVerifiedPieceStorage({manifestBytes,indexBytes:changed,indexIdentity:sha(changed),acquire:async()=>data[0]}),/binding/);
console.log('verified-piece-storage: selective reads, corruption, binding and close passed');

// A host retains its ArrayBuffer; changing it after hashing cannot change returned bytes.
const mutable = data[0].slice().buffer;
const owned = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity, acquire: async () => mutable });
const originalDigest = crypto.subtle.digest;
crypto.subtle.digest = async function (...arguments_) {
  const result = await originalDigest.apply(this, arguments_);
  new Uint8Array(mutable).fill(9);
  return result;
};
try { assert.deepEqual([...new Uint8Array(await owned.storage.loadShardRange(0, 0, 4))], [1, 2, 3, 4]); }
finally { crypto.subtle.digest = originalDigest; await owned.storage.close(); }
for (const payload of [[1, 2, 3, 4], new Uint8Array(8), '1234']) {
  const invalid = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity, acquire: async () => payload });
  await assert.rejects(invalid.storage.loadShardRange(0, 0, 4), /integrity/);
  await invalid.storage.close();
}

const mutableIndex = indexBytes.slice(), mutableManifest = manifestBytes.slice();
crypto.subtle.digest = async function (...arguments_) {
  const result = await originalDigest.apply(this, arguments_);
  mutableIndex.fill(32); mutableManifest.fill(32);
  return result;
};
let metadataOwned;
try {
  metadataOwned = await createVerifiedPieceStorage({ manifestBytes: mutableManifest, indexBytes: mutableIndex,
    indexIdentity, acquire: async () => data[0] });
} finally { crypto.subtle.digest = originalDigest; }
assert.deepEqual([...new Uint8Array(await metadataOwned.storage.loadShardRange(0, 0, 4))], [1, 2, 3, 4]);
await metadataOwned.storage.close();

// Closing the public adapter settles owned range wrappers even if a host ignores abort.
let acquisitionStarted;
const acquiring = new Promise(resolve => { acquisitionStarted = resolve; });
const never = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity,
  acquire: async () => { acquisitionStarted(); return new Promise(() => {}); } });
let readSettled = false;
const pending = never.storage.loadShardRange(0, 0, 1).then(
  () => { throw new Error('Uncooperative acquisition must not complete'); },
  error => { readSettled = true; return error; });
await acquiring;
await never.storage.close();
assert.equal(never.getReceipt().activeReadBytes, 0);
assert.equal(readSettled, true, 'close resolves only after the public read rejects');
assert.match((await pending).message, /closed/);
