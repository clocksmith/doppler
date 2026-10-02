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
