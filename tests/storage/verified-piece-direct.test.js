import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdtemp, rm, utimes, writeFile } from 'node:fs/promises';
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
