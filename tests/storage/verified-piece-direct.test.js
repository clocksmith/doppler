import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdtemp, open as openFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';

// Real files, real gather lanes and the real verification cache; no model and no GPU.
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const PIECE = 64, COUNT = 8;

function model() {
  const data = new Uint8Array(COUNT * PIECE), pieces = [];
  for (let i = 0; i < COUNT; i++) {
    const slice = data.subarray(i * PIECE, (i + 1) * PIECE);
    for (let j = 0; j < PIECE; j++) slice[j] = (i * 37 + j * 13 + 5) & 0xff;
    pieces.push({ offset: i * PIECE, size: PIECE, identity: sha(slice) });
  }
  const manifestBytes = encode({ hashAlgorithm: 'sha256',
    shards: [{ filename: 'experts.bin', path: 'experts.bin', size: data.byteLength, hash: sha(data).slice(7), hashAlgorithm: 'sha256' }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: PIECE,
    files: [{ path: 'experts.bin', size: data.byteLength, pieces }] });
  return { data, manifestBytes, indexBytes, indexIdentity: sha(indexBytes) };
}

async function open(root, { bindIdentities = true, cache = true } = {}) {
  const m = model(), file = join(root, 'experts.bin');
  await writeFile(file, m.data);
  const st = await lstat(file, { bigint: true });
  const opened = await createVerifiedPieceStorage({ manifestBytes: m.manifestBytes, indexBytes: m.indexBytes, indexIdentity: m.indexIdentity,
    acquire: async piece => m.data.subarray(piece.offset, piece.offset + piece.size),
    nodeGather: { files: { 'experts.bin': file }, maxPartBytes: 4 * PIECE, maxRanges: 3, memoryBudgetBytes: 2 * (4 * PIECE + 2 * PIECE), concurrency: 2,
      ...(bindIdentities ? { fileIdentities: { 'experts.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() } } } : {}),
      ...(cache ? { verificationCache: { directory: join(root, 'state') } } : {}) } });
  return { ...m, file, opened, port: opened.storage.rangeLease };
}
const signal = () => new AbortController().signal;
async function verify(port, offset, length) { const lease = await port.read(0, offset, length, signal()); lease.release(); }

test('direct verified reads serve only ranges whose every piece is verified for the current file state', async () => {
  const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
  let f;
  try {
    f = await open(root);
    assert.equal(typeof f.port.readVerifiedInto, 'function');
    const target = new Uint8Array(100).fill(0xee);
    // Nothing is verified yet: declined, and the lane path remains the only way in.
    assert.equal(await f.port.readVerifiedInto(0, 70, target, signal()), false);
    // A lane verifies pieces 1 and 2 (bytes 64..191). The range 70..169 lies inside them.
    await verify(f.port, 64, 2 * PIECE);
    assert.equal(await f.port.readVerifiedInto(0, 70, target, signal()), true);
    assert.deepEqual([...target], [...f.data.subarray(70, 170)]);
    // One byte past the verified pieces reaches an unverified piece: declined as a whole.
    assert.equal(await f.port.readVerifiedInto(0, 100, new Uint8Array(100), signal()), false);
    assert.equal(await f.port.readVerifiedInto(0, 0, new Uint8Array(70), signal()), false);
    // Bounds and arguments.
    await assert.rejects(f.port.readVerifiedInto(0, 64, new Uint8Array(0), signal()), /non-empty destination/);
    await assert.rejects(f.port.readVerifiedInto(7, 64, new Uint8Array(8), signal()), /Undeclared gather shard/);
    assert.equal(await f.port.readVerifiedInto(0, COUNT * PIECE - 8, new Uint8Array(16), signal()), false);
    await assert.rejects(f.port.readVerifiedInto(0, 64, new Uint8Array(8), AbortSignal.abort(new Error('stop'))), /stop/);
    const receipt = await f.opened.getGatherReceipt();
    assert.deepEqual([receipt.directVerifiedReads, receipt.directVerifiedBytes], [1, 100]);
    assert.equal(receipt.directDeclinedReads, 4);
    // The file's state changes (same bytes, new mtime): every verified bit is void for direct reads.
    const later = new Date(Date.now() + 5000);
    await utimes(f.file, later, later);
    assert.equal(await f.port.readVerifiedInto(0, 70, new Uint8Array(100), signal()), false);
    await f.opened.storage.close();
    await assert.rejects(f.port.readVerifiedInto(0, 70, new Uint8Array(100), signal()), /closed/);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('direct verified reads exist only with bound file identities and a verification cache', async () => {
  for (const options of [{ bindIdentities: false }, { cache: false }]) {
    const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
    let f;
    try {
      f = await open(root, options);
      assert.equal(f.port.readVerifiedInto, undefined);
    } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
  }
});

test('a file replaced after its pieces were verified is declined, not served', async () => {
  const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
  let f;
  try {
    f = await open(root);
    await verify(f.port, 0, 2 * PIECE);
    assert.equal(await f.port.readVerifiedInto(0, 0, new Uint8Array(PIECE), signal()), true);
    // Same size, different bytes, new inode at the approved pathname.
    const forged = f.data.slice(); forged[3] ^= 0xff;
    await rm(f.file); await writeFile(f.file, forged);
    const target = new Uint8Array(PIECE);
    assert.equal(await f.port.readVerifiedInto(0, 0, target, signal()), false);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('a read ahead of verification is usable only after a lane verifies its pieces for the same file state', async () => {
  const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
  let f;
  try {
    f = await open(root);
    assert.equal(f.port.hasVerified(0, 70, 100), false);
    const target = new Uint8Array(100).fill(0xee);
    const confirm = await f.port.readAheadInto(0, 70, target, signal());
    assert.equal(typeof confirm, 'function');
    // The bytes are there, and they are not yet authentic.
    assert.deepEqual([...target], [...f.data.subarray(70, 170)]);
    assert.equal(confirm(), false);
    // Only one of the two covering pieces is verified: still not usable.
    await verify(f.port, 64, PIECE);
    assert.equal(confirm(), false);
    await verify(f.port, 128, PIECE);
    assert.equal(f.port.hasVerified(0, 70, 100), true);
    assert.equal(confirm(), true);
    // Already verified pieces resolve true directly, as readVerifiedInto does.
    assert.equal(await f.port.readAheadInto(0, 70, new Uint8Array(100), signal()), true);
    const receipt = await f.opened.getGatherReceipt();
    assert.deepEqual([receipt.directAheadReads, receipt.directAheadBytes, receipt.directAheadConfirmed], [1, 100, 1]);
    await assert.rejects(f.port.readAheadInto(7, 64, new Uint8Array(8), signal()), /Undeclared gather shard/);
    assert.equal(f.port.hasVerified(7, 64, 8), false);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('a read ahead is never confirmed when the file changes before its pieces are verified', async () => {
  // The file's state changes between the read and the verification (same bytes, new mtime).
  let root = await mkdtemp(join(tmpdir(), 'direct-verified-')), f;
  try {
    f = await open(root);
    const confirm = await f.port.readAheadInto(0, 0, new Uint8Array(PIECE), signal());
    const later = new Date(Date.now() + 5000);
    await utimes(f.file, later, later);
    await verify(f.port, 0, PIECE);
    assert.equal(confirm(), false);
    assert.equal(f.port.hasVerified(0, 0, PIECE), false);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }

  // The file is replaced with forged bytes at the approved pathname after the read.
  root = await mkdtemp(join(tmpdir(), 'direct-verified-')); f = undefined;
  try {
    f = await open(root);
    const confirm = await f.port.readAheadInto(0, 0, new Uint8Array(PIECE), signal());
    const forged = f.data.slice(); forged[3] ^= 0xff;
    await rm(f.file); await writeFile(f.file, forged);
    await verify(f.port, 0, PIECE).catch(() => {});
    assert.equal(confirm(), false);
    // A file that is already not the admitted one is not read at all.
    assert.equal(await f.port.readAheadInto(0, PIECE, new Uint8Array(PIECE), signal()), false);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }

  // Forged bytes under the admitted file state: the lane's hash fails, so nothing is marked and nothing is confirmed.
  root = await mkdtemp(join(tmpdir(), 'direct-verified-')); f = undefined;
  try {
    const m = model(), file = join(root, 'experts.bin');
    const forged = m.data.slice(); forged[3] ^= 0xff;
    await writeFile(file, forged);
    const st = await lstat(file, { bigint: true });
    const opened = await createVerifiedPieceStorage({ manifestBytes: m.manifestBytes, indexBytes: m.indexBytes, indexIdentity: m.indexIdentity,
      acquire: async piece => m.data.subarray(piece.offset, piece.offset + piece.size),
      nodeGather: { files: { 'experts.bin': file }, maxPartBytes: 4 * PIECE, maxRanges: 3, memoryBudgetBytes: 2 * (4 * PIECE + 2 * PIECE), concurrency: 2,
        fileIdentities: { 'experts.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() } },
        verificationCache: { directory: join(root, 'state') } } });
    f = { opened };
    const port = opened.storage.rangeLease, target = new Uint8Array(PIECE);
    const confirm = await port.readAheadInto(0, 0, target, signal());
    assert.equal(target[3], forged[3], 'the forged byte was read into caller memory');
    await assert.rejects(verify(port, 0, PIECE));
    assert.equal(confirm(), false);
    assert.equal(port.hasVerified(0, 0, PIECE), false);
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('background verification waits for a lane; foreground reads keep their bound and go first', async () => {
  const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
  let f;
  try {
    f = await open(root);   // two lanes
    const range = index => [{ shard: 0, offset: index * PIECE, length: PIECE }];
    // Two foreground leases hold both lanes. A third foreground read is still the caller's error.
    const first = await f.port.readMany(range(0), signal()), second = await f.port.readMany(range(1), signal());
    await assert.rejects(f.port.readMany(range(2), signal()), /Release a current gather lease/);
    // Background jobs queue instead, and run as lanes come free.
    const order = [];
    const queued = [2, 3, 4].map(index => f.port.verifyMany(range(index)).then(timings => { order.push(index); return timings; }));
    await new Promise(resolve => setTimeout(resolve, 10));
    assert.deepEqual(order, []);
    assert.equal(f.port.hasVerified(0, 2 * PIECE, PIECE), false);
    first.release(); second.release();
    const timings = await Promise.all(queued);
    assert.deepEqual(order.slice().sort(), [2, 3, 4]);
    assert(timings.every(timing => timing === null || typeof timing.verificationMs === 'number'));
    assert.equal(f.port.hasVerified(0, 2 * PIECE, 3 * PIECE), true);
    // A queued background job can be abandoned, and bad arguments are refused.
    const held = [await f.port.readMany(range(5), signal()), await f.port.readMany(range(6), signal())];
    const abandon = new AbortController();
    const waiting = f.port.verifyMany(range(7), abandon.signal);
    abandon.abort(new Error('stop'));
    await assert.rejects(waiting, /stop/);
    for (const lease of held) lease.release();
    await assert.rejects(f.port.verifyMany([]), /admitted ranges bound/);
    await assert.rejects(f.port.verifyMany([{ shard: 9, offset: 0, length: PIECE }]), /Undeclared gather shard/);
    // A foreground read that arrives while background jobs hold every lane waits, and is served before
    // background jobs that were queued ahead of it.
    const gate = [await f.port.readMany(range(0), signal()), await f.port.readMany(range(1), signal())];
    const served = [];
    const background = [5, 6, 7].map(index => f.port.verifyMany(range(index)).then(() => served.push('background-' + index)));
    for (const lease of gate) lease.release();   // the first two background jobs now hold both lanes
    const foreground = f.port.readMany(range(2), signal()).then(lease => { served.push('foreground'); lease.release(); });
    await Promise.all([...background, foreground]);
    assert(served.indexOf('foreground') < served.indexOf('background-7'), `foreground before the queued background job: ${served}`);
    // Jobs still queued when the pool closes are rejected. Both lanes stay held, so the job cannot start.
    const last = [await f.port.readMany(range(0), signal()), await f.port.readMany(range(1), signal())];
    const stranded = assert.rejects(f.port.verifyMany(range(7)), /closed/);
    await f.opened.storage.close();
    await stranded;
    for (const lease of last) lease.release();
  } finally { await f?.opened.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('authenticated bytes from the protected cache do not make the file itself verified', async () => {
  const root = await mkdtemp(join(tmpdir(), 'direct-verified-'));
  let session;
  try {
    const m = model(), file = join(root, 'experts.bin'), cache = join(root, 'protected');
    await writeFile(file, m.data);
    const st = await lstat(file, { bigint: true });
    const start = extra => createVerifiedPieceStorage({ manifestBytes: m.manifestBytes, indexBytes: m.indexBytes, indexIdentity: m.indexIdentity,
      acquire: async piece => m.data.subarray(piece.offset, piece.offset + piece.size),
      nodeGather: { files: { 'experts.bin': file }, maxPartBytes: 4 * PIECE, maxRanges: 3, memoryBudgetBytes: 2 * (4 * PIECE + 2 * PIECE), concurrency: 2,
        fileIdentities: { 'experts.bin': { device: st.dev.toString(), inode: st.ino.toString(), size: st.size.toString() } },
        protectedCache: { directory: cache, maxBytes: 6 * PIECE, staleAfterMs: 60_000 }, ...extra } });
    // First session, protected cache only: piece 0 is hashed from the file and kept as a protected copy.
    session = await start({});
    await verify(session.storage.rangeLease, 0, PIECE);
    assert.equal((await session.getGatherReceipt()).protectedCacheAdmissions, 1);
    await session.storage.close();
    // The file is altered in place at piece 0: same inode and size, new contents.
    const handle = await openFile(file, 'r+');
    await handle.write(Uint8Array.of(m.data[3] ^ 0xff), 0, 1, 3);
    await handle.close();
    // Second session, with a verification cache as well. The lane serves piece 0 from the protected copy.
    session = await start({ verificationCache: { directory: join(root, 'state') } });
    const port = session.storage.rangeLease;
    const lease = await port.read(0, 0, PIECE, signal());
    assert.deepEqual([...lease.bytes], [...m.data.subarray(0, PIECE)], 'the lease holds the genuine bytes');
    lease.release();
    assert.equal((await session.getGatherReceipt()).protectedCacheHits, 1);
    // The file's own bytes at piece 0 were never hashed, so nothing may read them directly.
    assert.equal(port.hasVerified(0, 0, PIECE), false);
    const target = new Uint8Array(PIECE);
    assert.equal(await port.readVerifiedInto(0, 0, target, signal()), false);
    const ahead = await port.readAheadInto(0, 0, target, signal());
    assert.equal(typeof ahead, 'function');
    assert.equal(ahead(), false);
  } finally { await session?.storage.close().catch(() => {}); await rm(root, { recursive: true, force: true }); }
});
