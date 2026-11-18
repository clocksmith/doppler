import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, symlink, unlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

const sha = x => 'sha256:' + createHash('sha256').update(x).digest('hex');
const enc = x => new TextEncoder().encode(JSON.stringify(x));
const good = new Uint8Array(64).fill(1), bad = new Uint8Array(64).fill(9);
const manifestBytes = enc({ modelId: 'test', hashAlgorithm: 'sha256', shards: [{ filename: 'weights.bin', size: 64, hash: sha(good).slice(7) }] });
const indexBytes = enc({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 64, files: [{ path: 'weights.bin', size: 64, pieces: [{ offset: 0, size: 64, identity: sha(good) }] }] });
const metadata = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes) };

test('generic acquire cannot inherit verified local piece state', async () => {
  const root = await mkdtemp(join(tmpdir(), 'acquire-inherit-'));
  let source;
  try {
    const pieces = join(root, 'pieces');
    await mkdir(pieces);
    await writeFile(join(pieces, 'weights.bin.0'), good);
    let corrupt = false;
    source = await createVerifiedPieceStorage({
      ...metadata,
      pieceDirectoryVerificationCache: { directory: join(root, 'state'), pieceDirectory: pieces },
      acquire: async () => corrupt ? bad : good,
    });
    assert.deepEqual(new Uint8Array(await source.storage.loadShardRange(0, 0, 64)), good);
    assert.deepEqual(new Uint8Array(await source.storage.loadShardRange(0, 0, 64)), good);
    corrupt = true;
    await assert.rejects(source.storage.loadShardRange(0, 0, 64), /integrity mismatch/);
  } finally {
    await source?.storage.close();
    await rm(root, { recursive: true, force: true });
  }
});

test('generic acquire cannot seed worker authentication for unrelated file bytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'acquire-seed-'));
  let source;
  try {
    const file = join(root, 'weights.bin');
    await writeFile(file, bad);
    source = await createVerifiedPieceStorage({
      ...metadata,
      acquire: async () => good,
      nodeGather: {
        files: { 'weights.bin': file },
        maxPartBytes: 64,
        maxRanges: 1,
        memoryBudgetBytes: 256,
        concurrency: 1,
        verificationCache: { directory: join(root, 'state') },
      },
    });
    assert.deepEqual(new Uint8Array(await source.storage.loadShardRange(0, 0, 64)), good);
    await assert.rejects(source.storage.rangeLease.read(0, 0, 64, new AbortController().signal), /integrity mismatch/);
  } finally {
    await source?.storage.close();
    await rm(root, { recursive: true, force: true });
  }
});

test('owned nodePieceDirectory binds state verification to the opened regular file descriptor and rejects callbacks, mutations and symlinks', async () => {
  const root = await mkdtemp(join(tmpdir(), 'owned-piece-dir-'));
  let first, second, third;
  try {
    const pieces = join(root, 'pieces');
    const state = join(root, 'state');
    await mkdir(pieces);
    const target = join(pieces, 'weights.bin.0');
    await writeFile(target, good);

    await assert.rejects(
      createVerifiedPieceStorage({
        ...metadata,
        acquire: async () => bad,
        nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: { directory: state } },
      }),
      /cannot be combined/,
    );

    first = await createVerifiedPieceStorage({
      ...metadata,
      nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: { directory: state } },
    });
    assert.deepEqual(new Uint8Array(await first.storage.loadShardRange(0, 0, 64)), good);
    assert.equal(first.getReceipt().stateCacheHits ?? 0, 0);
    await first.storage.close();
    first = null;

    second = await createVerifiedPieceStorage({
      ...metadata,
      nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: { directory: state } },
    });
    assert.deepEqual(new Uint8Array(await second.storage.loadShardRange(0, 0, 64)), good);
    assert.equal(second.getReceipt().stateCacheHits, 1);

    await writeFile(target, bad);
    await assert.rejects(second.storage.loadShardRange(0, 0, 64), /integrity mismatch/);

    const external = join(root, 'outside.bin');
    await writeFile(external, good);
    await unlink(target);
    await symlink(external, target);
    third = await createVerifiedPieceStorage({
      ...metadata,
      nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: { directory: state } },
    });
    await assert.rejects(third.storage.loadShardRange(0, 0, 64));
  } finally {
    await first?.storage.close();
    await second?.storage.close();
    await third?.storage.close();
    await rm(root, { recursive: true, force: true });
  }
});


test('owned piece FIFO rejects and cleanup settles without a writer', { skip: process.platform === 'win32', timeout: 2000 }, async () => {
 const root = await mkdtemp(join(tmpdir(), 'piece-fifo-'));
 let source;
 try {
  execFileSync('mkfifo', [join(root, 'weights.bin.0')]);
  source = await createVerifiedPieceStorage({...metadata, nodePieceDirectory: { directory: root, maxPieceBytes: 64, maxActiveReadBytes: 64 }});
  await assert.rejects(source.storage.loadShardRange(0,0,64), /regular file/);
  await source.storage.close();
 } finally { await source?.storage.close(); await rm(root, {recursive:true,force:true}); }
});
