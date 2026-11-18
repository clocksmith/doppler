import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdir, mkdtemp, readdir, readFile, rm, symlink, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));

function makeModelPieces(count, pieceSize = 64) {
  const pieces = [];
  const data = new Uint8Array(count * pieceSize);
  for (let i = 0; i < count; i++) {
    const slice = data.subarray(i * pieceSize, (i + 1) * pieceSize);
    for (let j = 0; j < pieceSize; j++) slice[j] = (i * 41 + j * 17 + 11) & 0xff;
    pieces.push({ offset: i * pieceSize, size: pieceSize, identity: sha(slice) });
  }
  const manifestBytes = encode({
    hashAlgorithm: 'sha256',
    shards: [{ filename: 'experts.bin', path: 'experts.bin', size: data.byteLength, hash: sha(data).slice(7), hashAlgorithm: 'sha256' }],
  });
  const indexBytes = encode({
    schema: 'doppler.verified-pieces/v1',
    manifestIdentity: sha(manifestBytes),
    maxPieceBytes: pieceSize,
    files: [{ path: 'experts.bin', size: data.byteLength, pieces }],
  });
  return { data, pieces, manifestBytes, indexBytes, indexIdentity: sha(indexBytes) };
}

test('per-file-state verification cache skips redundant critical-path hashing and invalidates on any file mutation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gather-state-cache-'));
  try {
    const { data, manifestBytes, indexBytes, indexIdentity } = makeModelPieces(6, 64);
    const weightPath = join(root, 'experts.bin');
    const stateDir = join(root, 'state-cache');
    await writeFile(weightPath, data);
    const st = await lstat(weightPath, { bigint: true });
    const fileIdentities = {
      'experts.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() },
    };

    const acquirePiece = async piece => {
      const full = await readFile(weightPath);
      return full.subarray(piece.offset, piece.offset + piece.size);
    };

    const openStorage = (extra = {}) => createVerifiedPieceStorage({
      manifestBytes,
      indexBytes,
      indexIdentity,
      acquire: acquirePiece,
      nodeGather: {
        files: { 'experts.bin': weightPath },
        fileIdentities,
        maxPartBytes: 192,
        maxRanges: 3,
        memoryBudgetBytes: 2 * (192 + 128),
        concurrency: 2,
        verificationCache: { directory: stateDir, ...extra },
      },
    });

    // Pass 1 (unverified file state): authenticates via SHA-256 and records bits on close.
    const first = await openStorage();
    assert.equal(first.storage.rangeLease.verificationCacheBound, true);
    const denseBuf = await first.storage.loadShardRange(0, 0, 64);
    assert.deepEqual([...new Uint8Array(denseBuf)], [...data.subarray(0, 64)]);
    // Generic callback bytes do not authenticate the local file; the worker hashes its own FD read.
    const l0 = await first.storage.rangeLease.read(0, 0, 64, new AbortController().signal);
    assert.deepEqual([...l0.bytes], [...data.subarray(0, 64)]);
    l0.release();
    const b1 = await first.storage.rangeLease.readMany([
      { shard: 0, offset: 64, length: 64 },
      { shard: 0, offset: 128, length: 64 },
      { shard: 0, offset: 192, length: 64 },
    ], new AbortController().signal);
    assert.deepEqual([...b1.bytes], [...data.subarray(64, 256)]);
    b1.release();
    const p1 = await first.storage.rangeLease.read(0, 260, 40, new AbortController().signal);
    assert.deepEqual([...p1.bytes], [...data.subarray(260, 300)]);
    p1.release();
    const r1 = await first.getGatherReceipt();
    assert.equal(r1.stateCacheHits, 0);
    assert.equal(r1.stateCacheMisses, 5);
    assert.equal(r1.stateCacheAdmissions, 5);
    await first.storage.close();

    // Pass 2 (warm file state across fresh open): zero critical-path SHA-256 on verified pieces.
    const second = await openStorage();
    const denseBuf2 = await second.storage.loadShardRange(0, 0, 64);
    assert.deepEqual([...new Uint8Array(denseBuf2)], [...data.subarray(0, 64)]);
    assert.equal(second.getReceipt().stateCacheHits ?? 0, 0);
    const b2 = await second.storage.rangeLease.readMany([
      { shard: 0, offset: 0, length: 64 },
      { shard: 0, offset: 64, length: 64 },
      { shard: 0, offset: 128, length: 64 },
    ], new AbortController().signal);
    assert.deepEqual([...b2.bytes], [...data.subarray(0, 192)]);
    assert.equal(b2.timings.verificationMs, 0);
    b2.release();
    const p2 = await second.storage.rangeLease.read(0, 260, 40, new AbortController().signal);
    assert.deepEqual([...p2.bytes], [...data.subarray(260, 300)]);
    assert.equal(p2.timings.verificationMs, 0);
    p2.release();
    const r2 = await second.getGatherReceipt();
    assert.equal(r2.stateCacheHits, 4);
    assert.equal(r2.stateCacheMisses, 0);
    assert.equal(r2.verificationMs, 0);

    // In-place corruption (same inode + size, updated mtimeNs/ctimeNs) invalidates cached bits and rejects!
    const corrupted = new Uint8Array(data);
    corrupted[70] ^= 0xff;
    const future = new Date(Date.now() + 5000);
    await writeFile(weightPath, corrupted);
    await utimes(weightPath, future, future);
    await assert.rejects(
      second.storage.rangeLease.read(0, 64, 64, new AbortController().signal),
      /integrity mismatch/,
    );
    await second.storage.close();

    // Corrupting the persisted .state file heals cleanly via fresh SHA-256 on reopen.
    await writeFile(weightPath, data);
    const restoredStat = await lstat(weightPath, { bigint: true });
    fileIdentities['experts.bin'] = {
      device: restoredStat.dev.toString(),
      inode: restoredStat.ino.toString(),
      size: restoredStat.size.toString(),
    };
    const third = await openStorage();
    const heal1 = await third.storage.rangeLease.read(0, 0, 64, new AbortController().signal);
    heal1.release();
    await third.storage.close();
    for (const entry of await readdir(stateDir)) {
      if (entry.endsWith('.state')) {
        await writeFile(join(stateDir, entry), new Uint8Array(36).fill(0xaa));
      }
    }
    const fourth = await openStorage();
    const heal2 = await fourth.storage.rangeLease.read(0, 0, 64, new AbortController().signal);
    assert.deepEqual([...heal2.bytes], [...data.subarray(0, 64)]);
    heal2.release();
    const r4 = await fourth.getGatherReceipt();
    assert.equal(r4.stateCacheMisses, 1);
    await fourth.storage.close();

    // Symlink state directories are rejected.
    const symlinkDir = join(root, 'state-link');
    await symlink(stateDir, symlinkDir);
    await assert.rejects(openStorage({ directory: symlinkDir }), /non-symlink directory/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('idle background verification hashes remaining pieces off the critical path and piece-directory state cache validates per-piece files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gather-bg-state-'));
  try {
    const { data, pieces, manifestBytes, indexBytes, indexIdentity } = makeModelPieces(8, 64);
    const weightPath = join(root, 'experts.bin');
    const stateDir = join(root, 'bg-state');
    await writeFile(weightPath, data);

    const bgOpened = await createVerifiedPieceStorage({
      manifestBytes,
      indexBytes,
      indexIdentity,
      acquire: async () => { throw new Error('direct gather expected'); },
      nodeGather: {
        files: { 'experts.bin': weightPath },
        maxPartBytes: 64,
        maxRanges: 3,
        memoryBudgetBytes: 2 * (64 + 128),
        concurrency: 2,
        verificationCache: { directory: stateDir, backgroundVerify: true, backgroundIdleDelayMs: 5 },
      },
    });
    await new Promise(resolve => setTimeout(resolve, 120));
    const lease = await bgOpened.storage.rangeLease.read(0, 7 * 64, 64, new AbortController().signal);
    assert.deepEqual([...lease.bytes], [...data.subarray(7 * 64, 8 * 64)]);
    assert.equal(lease.timings.verificationMs, 0);
    lease.release();
    const bgReceipt = await bgOpened.getGatherReceipt();
    assert(bgReceipt.backgroundVerifiedPieces >= 1);
    assert.equal(bgReceipt.stateCacheHits, 1);
    await bgOpened.storage.close();

    // Per-piece directory state cache (Stage 1 partial piece cache mode).
    const pieceDir = join(root, 'pieces');
    const pieceStateDir = join(root, 'piece-state');
    await mkdir(pieceDir, { recursive: true });
    for (const p of pieces) {
      await writeFile(join(pieceDir, `experts.bin.${p.offset}`), data.subarray(p.offset, p.offset + p.size));
    }
    const acquireFromDir = async piece => readFile(join(pieceDir, `${piece.path}.${piece.offset}`));
    const pOpen = () => createVerifiedPieceStorage({
      manifestBytes,
      indexBytes,
      indexIdentity,
      acquire: acquireFromDir,
      pieceDirectoryVerificationCache: { directory: pieceStateDir, pieceDirectory: pieceDir },
    });

    const p1 = await pOpen();
    const out1 = await p1.storage.loadShardRange(0, 0, 128);
    assert.deepEqual([...new Uint8Array(out1)], [...data.subarray(0, 128)]);
    assert.equal(p1.getReceipt().verifiedBytes, 128);
    await p1.storage.close();

    const p2 = await pOpen();
    const out2 = await p2.storage.loadShardRange(0, 0, 128);
    assert.deepEqual([...new Uint8Array(out2)], [...data.subarray(0, 128)]);
    assert.equal(p2.getReceipt().stateCacheHits ?? 0, 0);
    assert.equal(p2.getReceipt().verifiedBytes, 128);
    await p2.storage.close();

    // Mutating a single piece file invalidates its slot and catches corruption.
    const badPiece = new Uint8Array(64).fill(0x55);
    const future = new Date(Date.now() + 5000);
    const targetPieceFile = join(pieceDir, 'experts.bin.64');
    await writeFile(targetPieceFile, badPiece);
    await utimes(targetPieceFile, future, future);
    const p3 = await pOpen();
    await assert.rejects(p3.storage.loadShardRange(0, 64, 64), /integrity mismatch/);
    await p3.storage.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
