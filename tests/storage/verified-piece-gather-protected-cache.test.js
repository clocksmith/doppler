import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdtemp, readdir, rm, symlink, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));

function makePieces(count, pieceSize = 64) {
  const pieces = [];
  const data = new Uint8Array(count * pieceSize);
  for (let i = 0; i < count; i++) {
    const slice = data.subarray(i * pieceSize, (i + 1) * pieceSize);
    for (let j = 0; j < pieceSize; j++) slice[j] = (i * 37 + j * 13 + 7) & 0xff;
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

test('protected piece cache serves SHA-verified hits across fresh gather opens and heals corrupted cache files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gather-protected-cache-'));
  try {
    const { data, pieces, manifestBytes, indexBytes, indexIdentity } = makePieces(6, 64);
    const weightPath = join(root, 'experts.bin');
    const cacheDir = join(root, 'protected-cache');
    await writeFile(weightPath, data);
    const st = await lstat(weightPath, { bigint: true });
    const fileIdentities = {
      'experts.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() },
    };

    const openStorage = () => createVerifiedPieceStorage({
      manifestBytes,
      indexBytes,
      indexIdentity,
      acquire: async () => { throw new Error('direct gather path expected'); },
      nodeGather: {
        files: { 'experts.bin': weightPath },
        fileIdentities,
        maxPartBytes: 192,
        maxRanges: 3,
        memoryBudgetBytes: 4 * (192 + 128),
        concurrency: 4,
        protectedCache: { directory: cacheDir, maxBytes: 6 * 64, staleAfterMs: 60_000 },
      },
    });

    // Pass 1 (cold): populates protected cache across single, batch, and partial reads.
    const first = await openStorage();
    assert.equal(first.storage.rangeLease.protectedCacheBound, true);
    const s1 = await first.storage.rangeLease.read(0, 0, 64, new AbortController().signal);
    assert.deepEqual([...s1.bytes], [...data.subarray(0, 64)]);
    s1.release();
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
    const receipt1 = await first.getGatherReceipt();
    assert.equal(receipt1.protectedCacheHits, 0);
    assert.equal(receipt1.protectedCacheMisses, 5);
    assert.equal(receipt1.protectedCacheAdmissions, 5);
    await first.storage.close();

    // Pass 2 (warm, fresh storage instance): 100% cache hits, exact bytes.
    const second = await openStorage();
    const b2 = await second.storage.rangeLease.readMany([
      { shard: 0, offset: 0, length: 64 },
      { shard: 0, offset: 64, length: 64 },
      { shard: 0, offset: 128, length: 64 },
    ], new AbortController().signal);
    assert.deepEqual([...b2.bytes], [...data.subarray(0, 192)]);
    b2.release();
    const p2 = await second.storage.rangeLease.read(0, 260, 40, new AbortController().signal);
    assert.deepEqual([...p2.bytes], [...data.subarray(260, 300)]);
    p2.release();
    const receipt2 = await second.getGatherReceipt();
    assert.equal(receipt2.protectedCacheHits, 4);
    assert.equal(receipt2.protectedCacheMisses, 0);

    // Corrupting a cached .piece file is detected by SHA-256, unlinked, and healed from the approved file.
    const targetHex = pieces[0].identity.slice(7);
    await writeFile(join(cacheDir, '00', `${targetHex}.piece`), new Uint8Array(64).fill(0xee));
    const healed = await second.storage.rangeLease.read(0, 0, 64, new AbortController().signal);
    assert.deepEqual([...healed.bytes], [...data.subarray(0, 64)]);
    healed.release();
    await second.storage.close();

    // Underlying source file identity is still enforced even when every piece is warm in the protected cache.
    const third = await openStorage();
    const replacement = join(root, 'experts.swap');
    await writeFile(replacement, data);
    await rm(weightPath);
    await writeFile(weightPath, data);
    const swappedStat = await lstat(weightPath, { bigint: true });
    if (swappedStat.ino !== st.ino) {
      await assert.rejects(
        third.storage.rangeLease.read(0, 64, 64, new AbortController().signal),
        /identity changed/,
      );
    }
    await third.storage.close();

    // Symlink cache directories are rejected.
    const linkDir = join(root, 'cache-link');
    await symlink(cacheDir, linkDir);
    await assert.rejects(createVerifiedPieceGather({
      manifestBytes,
      indexBytes,
      indexIdentity,
      files: { 'experts.bin': replacement },
      maxPartBytes: 64,
      memoryBudgetBytes: 192,
      protectedCache: { directory: linkDir, maxBytes: 256 },
    }), /non-symlink directory/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('protected piece cache avoids cyclic scan thrashing and evicts stale entries within maxBytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gather-protected-scan-'));
  try {
    const { data, manifestBytes, indexBytes, indexIdentity } = makePieces(12, 64);
    const weightPath = join(root, 'experts.bin');
    const cacheDir = join(root, 'scan-cache');
    await writeFile(weightPath, data);

    const runScan = async (startPiece, endPiece, staleAfterMs = 60_000) => {
      const gather = await createVerifiedPieceGather({
        manifestBytes,
        indexBytes,
        indexIdentity,
        files: { 'experts.bin': weightPath },
        maxPartBytes: 64,
        memoryBudgetBytes: 2 * (64 + 128),
        concurrency: 2,
        protectedCache: { directory: cacheDir, maxBytes: 4 * 64, staleAfterMs },
      });
      try {
        for (let i = startPiece; i < endPiece; i++) {
          const lease = await gather.readLease('experts.bin', i * 64, 64);
          assert.deepEqual([...lease.bytes], [...data.subarray(i * 64, (i + 1) * 64)]);
          lease.release();
        }
        return await gather.getReceipt();
      } finally {
        await gather.close();
      }
    };

    // Scan 8 pieces (512 B) through a 4-piece (256 B) cache twice: Pass 2 must hit all 4 retained slots (zero scan thrashing).
    await runScan(0, 8);
    const secondPass = await runScan(0, 8);
    assert.equal(secondPass.protectedCacheHits, 4);
    assert.equal(secondPass.protectedCacheMisses, 4);

    // Backdate cached files past staleAfterMs so a new disjoint workload (pieces 8..12) reclaims all 4 slots cleanly.
    const bucketDir = join(cacheDir, '00');
    const past = new Date(Date.now() - 10_000);
    for (const file of await readdir(bucketDir)) {
      if (file.endsWith('.piece')) await utimes(join(bucketDir, file), past, past);
    }
    await runScan(8, 12, 1000);
    const newWorkloadWarm = await runScan(8, 12, 60_000);
    assert.equal(newWorkloadWarm.protectedCacheHits, 4);

    // Total retained bytes on disk never exceed maxBytes (256 B).
    let retainedBytes = 0;
    for (const file of await readdir(bucketDir)) {
      if (file.endsWith('.piece')) retainedBytes += (await lstat(join(bucketDir, file))).size;
    }
    assert(retainedBytes <= 4 * 64, `expected retainedBytes <= 256, got ${retainedBytes}`);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
