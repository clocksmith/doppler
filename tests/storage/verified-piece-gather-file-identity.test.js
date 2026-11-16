import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

test('explicit worker file identities reject swaps before and after descriptor admission', { timeout: 5000 }, async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'gather-file-id-'));
  const filename = path.join(root, 'weights.bin'), retained = path.join(root, 'retained.bin');
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
  const encode = value => new TextEncoder().encode(JSON.stringify(value));
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes) }] }] });
  let gather;
  try {
    await fs.writeFile(filename, bytes);
    const stat = await fs.lstat(filename, { bigint: true });
    const identities = { 'weights.bin': { device: stat.dev.toString(), inode: stat.ino.toString(), size: stat.size.toString() } };
    const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files: { 'weights.bin': filename },
      fileIdentities: identities, maxPartBytes: 4, memoryBudgetBytes: 12 };
    await assert.rejects(createVerifiedPieceGather({ ...options, fileIdentities: {} }), /bind every/);
    await assert.rejects(createVerifiedPieceGather({ ...options, fileIdentities: null }), /bind every/);
    const pending = createVerifiedPieceGather(options);
    identities['weights.bin'].inode = '0';
    gather = await pending;
    assert.equal(gather.rangeLease.fileIdentitiesBound, true);
    // Matching contents cannot relabel a new inode as the approved source.
    await fs.rename(filename, retained); await fs.writeFile(filename, bytes);
    await assert.rejects(gather.readLease('weights.bin', 0, 4), /identity changed/);
    await fs.unlink(filename); await fs.rename(retained, filename);
    let lease = await gather.readLease('weights.bin', 0, 4);
    assert.deepEqual(lease.bytes, bytes); lease.release();
    await fs.rename(filename, retained); await fs.writeFile(filename, bytes);
    await assert.rejects(gather.readLease('weights.bin', 0, 4), /identity changed/);
    await fs.unlink(filename); await fs.symlink(retained, filename);
    await assert.rejects(gather.readLease('weights.bin', 0, 4), /identity changed/);
    await fs.unlink(filename); await fs.rename(retained, filename);
    // The identity check supplements rather than replaces piece authentication.
    await fs.writeFile(filename, Uint8Array.of(9, 2, 3, 4));
    await assert.rejects(gather.readLease('weights.bin', 0, 4), /integrity mismatch/);
    await fs.writeFile(filename, bytes);
    lease = await gather.readLease('weights.bin', 0, 4); assert.deepEqual(lease.bytes, bytes);
    await gather.close(); lease.release(); assert.equal(lease.bytes.byteLength, 0);
    await assert.rejects(gather.readLease('weights.bin', 0, 4), /closed/);
  } finally {
    await gather?.close(); await fs.rm(root, { recursive: true, force: true });
  }
});
