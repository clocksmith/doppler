import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const sha = b => 'sha256:' + createHash('sha256').update(b).digest('hex');
const encode = v => new TextEncoder().encode(JSON.stringify(v));
const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
const dir = await mkdtemp(join(tmpdir(), 'gather-'));
const path = join(dir, 'weights.bin');
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 8 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes),
  maxPieceBytes: 4, files: [{ path: 'weights.bin', size: 8, pieces: [0, 4].map(offset =>
    ({ offset, size: 4, identity: sha(bytes.subarray(offset, offset + 4)) })) }] });
const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files: { 'weights.bin': path },
  maxPartBytes: 8, memoryBudgetBytes: 16 };
let reader, other;
try {
  await writeFile(path, bytes);
  await assert.rejects(createVerifiedPieceGather({ ...options, memoryBudgetBytes: 15 }), /budget/);
  await assert.rejects(createVerifiedPieceGather({ ...options, indexIdentity: sha(bytes) }), /identity/);
  reader = await createVerifiedPieceGather(options);
  other = await createVerifiedPieceGather(options);
  const lease = await reader.readLease('weights.bin', 2, 5);
  assert.deepEqual([...lease.bytes], [3, 4, 5, 6, 7]);
  await assert.rejects(reader.readLease('weights.bin', 0, 4), /Release/);
  lease.release(); lease.release();
  assert.equal(lease.bytes.byteLength, 0);
  const again = await reader.readLease('weights.bin', 0, 8);
  assert.deepEqual([...again.bytes], [...bytes]);
  again.release();
  await assert.rejects(reader.readLease('weights.bin', 0, 9), /bound/);
  await writeFile(path, Uint8Array.of(1, 2, 3, 4, 0, 0, 0, 0));
  await assert.rejects(reader.readLease('weights.bin', 0, 8), /integrity/);
  await writeFile(path, bytes);
  const abort = new AbortController();
  const pending = reader.readLease('weights.bin', 0, 8, abort.signal);
  abort.abort(new Error('Cancelled read'));
  await assert.rejects(pending, /Cancelled/);
  const recovered = await reader.readLease('weights.bin', 0, 8);
  assert.deepEqual([...recovered.bytes], [...bytes]); recovered.release();
  await writeFile(path, bytes.subarray(0, 3));
  await assert.rejects(reader.readLease('weights.bin', 0, 8), /Incomplete/);
  await writeFile(path, bytes);
  const closing = reader.readLease('weights.bin', 0, 8);
  const rejected = assert.rejects(closing, /closed/);
  await reader.close(); await rejected;
  const isolated = await other.readLease('weights.bin', 0, 8);
  assert.deepEqual([...isolated.bytes], [...bytes]); isolated.release();
  await assert.rejects(reader.readLease('weights.bin', 0, 8), /closed/);
} finally { await reader?.close(); await other?.close(); await rm(dir, { recursive: true }); }
console.log('verified-piece-gather: authentication, bounds, leases, reuse, failure, cancellation and isolation passed');
