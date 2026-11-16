import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
const dir = await mkdtemp(join(tmpdir(), 'gather-concurrency-'));
const path = join(dir, 'weights.bin');
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 8 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: 8, pieces: [0, 4].map(offset =>
    ({ offset, size: 4, identity: sha(bytes.subarray(offset, offset + 4)) })) }] });
const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files: { 'weights.bin': path },
  maxPartBytes: 8, memoryBudgetBytes: 32, concurrency: 2 };
let reader;
try {
  await writeFile(path, bytes);
  await assert.rejects(createVerifiedPieceGather({ ...options, memoryBudgetBytes: 31 }), /budget/);
  await assert.rejects(createVerifiedPieceGather({ ...options, concurrency: 0 }), /budget/);
  reader = await createVerifiedPieceGather(options);
  assert.equal(reader.rangeLease.concurrency, 2);
  assert.equal(reader.rangeLease.memoryBytes, 32);
  const [a, b] = await Promise.all([reader.readLease('weights.bin', 0, 8), reader.readLease('weights.bin', 0, 8)]);
  assert.notEqual(a.bytes.buffer, b.bytes.buffer, 'Concurrent consumers never share transfer ownership');
  assert.deepEqual([...a.bytes], [...bytes]); assert.deepEqual([...b.bytes], [...bytes]);
  await assert.rejects(reader.readLease('weights.bin', 0, 4), /Release/);
  await assert.rejects(reader.getReceipt(), /Release/);
  a.release(); a.release();
  assert.equal(a.bytes.byteLength, 0); assert.equal(b.bytes.byteLength, 8);
  const c = await reader.readLease('weights.bin', 0, 4);
  b.release(); assert.equal(c.bytes.byteLength, 4); c.release();
  const receipt = await reader.getReceipt();
  assert.equal(receipt.requestedBytes, 20);
  assert.equal(receipt.verifiedBytes, 8, 'Authentication evidence deduplicates pieces across workers');
  assert.equal(receipt.ranges.length, 2);
  const cancel = new AbortController();
  const one = reader.readLease('weights.bin', 0, 4, cancel.signal);
  const two = reader.readLease('weights.bin', 4, 4);
  cancel.abort(new Error('One consumer cancelled'));
  await assert.rejects(one, /cancelled/);
  const surviving = await two; assert.deepEqual([...surviving.bytes], [5, 6, 7, 8]); surviving.release();
  await writeFile(path, Uint8Array.of(1, 2, 3, 4, 0, 0, 0, 0));
  const good = reader.readLease('weights.bin', 0, 4);
  const bad = reader.readLease('weights.bin', 4, 4);
  await assert.rejects(bad, /integrity/); (await good).release();
  await writeFile(path, bytes);
  const reads = [reader.readLease('weights.bin', 0, 8), reader.readLease('weights.bin', 0, 8)];
  const rejected = reads.map(read => assert.rejects(read, /closed/));
  await reader.close(); await Promise.all(rejected);
} finally { await reader?.close(); await rm(dir, { recursive: true }); }
console.log('verified-piece-gather-concurrency: independent leases, aggregate bounds, evidence, cancellation, corruption and close passed');
