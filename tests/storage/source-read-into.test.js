import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createNodeFileAccess, buildNodeFileReaders } from '../../src/tooling/source-runtime/node-loader.js';
import { createSourceStorageContext } from '../../src/storage/source-storage-context.js';
import { ShardCache } from '../../src/loader/shard-cache.js';
import { loadTensorRange } from '../../src/loader/tensors/tensor-reader.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-read-into-'));
const access = createNodeFileAccess();
try {
  const file = path.join(root, 'source.bin');
  const source = Uint8Array.from({ length: 64 }, (_, index) => index);
  await fs.writeFile(file, source);
  const destination = new Uint8Array(24).fill(255);
  assert.equal(await access.readInto(file, 5, destination.subarray(4, 20)), 16);
  assert.deepEqual(destination.slice(4, 20), source.slice(5, 21));
  assert(destination.slice(0, 4).every(byte => byte === 255));
  assert(destination.slice(20).every(byte => byte === 255));
  const readers = buildNodeFileReaders(access);
  const storage = createSourceStorageContext({ model: { shards: [{ size: 64 }] },
    shardSources: [{ path: file, size: 64 }], readRange: readers.readRange, readInto: readers.readInto,
    verifyHashes: false });
  const unverified = createSourceStorageContext({ model: { shards: [{ size: 64 }] },
    shardSources: [{ path: file, size: 64, hash: '0'.repeat(64), hashAlgorithm: 'sha256' }],
    readRange: readers.readRange, readInto: () => { throw new Error('must verify before filling'); }, verifyHashes: true });
  await assert.rejects(unverified.loadShardRangeInto(0, 0, new Uint8Array(4)), /hash mismatch/);
  const cache = new ShardCache({ maxEntries: 0, verifyHashes: false,
    customRangeLoader: storage.loadShardRange, customRangeIntoLoader: storage.loadShardRangeInto });
  const location = { shardIndex: 0, offset: 3, size: 40, dtype: 'U8', shape: [40] };
  let destinationBuffer;
  const result = await loadTensorRange(location, 'owned', 4, 16,
    () => { throw new Error('range allocation must not be used'); },
    async (shard, offset, view) => { destinationBuffer = view.buffer; return cache.loadRangeInto(shard, offset, view); });
  assert.equal(result.buffer, destinationBuffer, 'Node fills final tensor allocation directly');
  assert.deepEqual(result, source.slice(7, 23));
  const split = { size: 8, spans: [{ shardIndex: 0, offset: 2, size: 4 }, { shardIndex: 0, offset: 20, size: 4 }] };
  const splitResult = await loadTensorRange(split, 'split', 2, 5,
    () => { throw new Error('split must fill destination views'); },
    (shard, offset, view) => cache.loadRangeInto(shard, offset, view));
  assert.deepEqual([...splitResult], [4, 5, 20, 21, 22], 'span boundaries fill correct caller subviews');
  const borrowed = new Uint8Array([11, 12, 13, 14]);
  const fallback = new ShardCache({ maxEntries: 0, verifyHashes: false,
    customRangeLoader: async () => borrowed });
  const retained = new Uint8Array(4);
  await fallback.loadRangeInto(0, 0, retained);
  borrowed.fill(99);
  assert.deepEqual([...retained], [11, 12, 13, 14], 'borrowed callback storage cannot alias retained payload');
  const short = new ShardCache({ maxEntries: 0, verifyHashes: false, customRangeIntoLoader: async () => 2 });
  await assert.rejects(short.loadRangeInto(0, 0, new Uint8Array(4)), /short read/);
  await assert.rejects(loadTensorRange(location, 'short', 0, 4, async () => new ArrayBuffer(4), async () => 3), /returned 3 of 4/);
  await fs.truncate(file, 7);
  const tail = new Uint8Array(16).fill(255);
  assert.equal(await access.readInto(file, 0, tail), 7);
  assert(tail.slice(7).every(byte => byte === 255), 'short EOF never changes unfilled destination');
  await assert.rejects(storage.loadShardRangeInto(0, 0, new Uint8Array(16)), /short read/);
} finally { await access.close(); await fs.rm(root, { recursive: true, force: true }); }
console.log('readInto: exact caller ownership, Node/storage propagation, borrowed fallback, subview bounds and short EOF passed');
