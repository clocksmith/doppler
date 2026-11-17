import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));

async function fixture(signal) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'verified-storage-batch-'));
  const parts = [Uint8Array.of(1, 2, 3, 4), Uint8Array.of(5, 6, 7, 8)];
  const manifestBytes = encode({ shards: parts.map((bytes, i) => ({ filename: `${i}.bin`, size: bytes.length })) });
  const files = Object.fromEntries(parts.map((bytes, i) => [`${i}.bin`, path.join(root, `${i}.bin`)]));
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: parts.map((bytes, i) => ({ path: `${i}.bin`, size: bytes.length,
      pieces: [{ offset: 0, size: bytes.length, identity: sha(bytes) }] })) });
  await Promise.all(parts.map((bytes, i) => fs.writeFile(files[`${i}.bin`], bytes)));
  const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), signal,
    acquire: async piece => parts[Number(piece.path[0])],
    nodeGather: { files, maxPartBytes: 8, memoryBudgetBytes: 16 } };
  try {
    const source = await createVerifiedPieceStorage(options);
    return { source, options, cleanup: async () => { await source.storage.close(); await fs.rm(root, { recursive: true, force: true }); } };
  } catch (error) { await fs.rm(root, { recursive: true, force: true }); throw error; }
}

test('storage batch port preserves order, ownership and close admission', { timeout: 5000 }, async () => {
  const f = await fixture();
  try {
    const port = f.source.storage.rangeLease, signal = new AbortController().signal;
    assert.equal(typeof port.readMany, 'function');
    const lease = await port.readMany([{ shard: 1, offset: 1, length: 2 }, { shard: 0, offset: 0, length: 3 }], signal);
    assert(Object.isFrozen(lease.timings), 'Storage wrapper preserves the owned scalar timing snapshot');
    assert(Object.values(lease.timings).every(value => Number.isFinite(value) && value >= 0));
    assert.deepEqual([...lease.bytes], [6, 7, 1, 2, 3]);
    lease.release(); assert.equal(lease.bytes.byteLength, 0);
    assert.equal((await f.source.getGatherReceipt()).batchReads, 1);
    await f.source.storage.close();
    await assert.rejects(port.readMany([{ shard: 0, offset: 0, length: 1 }], signal), /closed/);
  } finally { await f.cleanup(); }
});

test('storage batch reads compose caller and source lifetime cancellation', { timeout: 5000 }, async () => {
  const lifetime = new AbortController(), f = await fixture(lifetime.signal);
  try {
    const port = f.source.storage.rangeLease, caller = new AbortController();
    caller.abort(new Error('caller cancelled batch'));
    await assert.rejects(port.readMany([{ shard: 0, offset: 0, length: 4 }], caller.signal), /caller cancelled/);
    const pending = port.readMany([{ shard: 0, offset: 0, length: 4 }, { shard: 1, offset: 0, length: 4 }], new AbortController().signal);
    const rejected = assert.rejects(pending, /source lifetime cancelled/);
    lifetime.abort(new Error('source lifetime cancelled')); await rejected;
    await f.source.storage.close();
    assert.equal(f.source.getReceipt().activeReadBytes, 0);
  } finally { await f.cleanup(); }
});

test('ordinary authenticated storage does not invent the optional batch capability', { timeout: 5000 }, async () => {
  const f = await fixture();
  let ordinary;
  try {
    ordinary = await createVerifiedPieceStorage({ ...f.options, nodeGather: undefined });
    assert.equal(typeof ordinary.storage.rangeLease?.readMany, 'undefined');
  } finally { await ordinary?.storage.close(); await f.cleanup(); }
});

test('storage close settles an in-flight batch before publishing a lease', { timeout: 5000 }, async () => {
  const f = await fixture();
  try {
    const pending = f.source.storage.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }], new AbortController().signal);
    const rejected = assert.rejects(pending, /closed/);
    await f.source.storage.close(); await rejected;
    assert.equal(f.source.getReceipt().activeReadBytes, 0);
  } finally { await f.cleanup(); }
});
