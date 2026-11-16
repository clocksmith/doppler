import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { createVerifiedPieceReader } from '../../src/storage/verified-piece-reader.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.from({ length: 11 }, (_, i) => i + 1);
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 11 }, { filename: 'other.bin', size: 4 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: 11, pieces: [0, 4, 8].map(offset => ({ offset,
    size: Math.min(4, 11 - offset), identity: sha(bytes.subarray(offset, offset + 4)) })) },
  { path: 'other.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes.subarray(0, 4)) }] }] });
const directory = await mkdtemp(join(tmpdir(), 'gather-direct-'));
let gathered, reference;
try {
  await writeFile(join(directory, 'weights.bin'), bytes);
  await writeFile(join(directory, 'other.bin'), bytes.subarray(0, 4));
  let acquired = 0;
  reference = await createVerifiedPieceReader({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    acquire: async piece => { acquired++; return bytes.slice(piece.offset, piece.offset + piece.size); } });
  const piece = reference.resolveCompletePiece('weights.bin', 8, 3);
  assert.deepEqual(piece, { path: 'weights.bin', offset: 8, size: 3, identity: sha(bytes.subarray(8)) });
  assert(Object.isFrozen(piece));
  assert.throws(() => { piece.offset = 0; }, TypeError);
  for (const [path, offset, length] of [['weights.bin', 0, 8], ['weights.bin', 1, 3], ['other.bin', 0, 3],
    ['missing.bin', 0, 4], ['weights.bin', -1, 4], ['weights.bin', 11, 1]]) {
    assert.equal(reference.resolveCompletePiece(path, offset, length), null);
  }
  assert.equal(acquired, 0, 'Metadata planning cannot claim or trigger byte verification');

  gathered = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    files: { 'weights.bin': join(directory, 'weights.bin'), 'other.bin': join(directory, 'other.bin') },
    maxPartBytes: 8, memoryBudgetBytes: 32, concurrency: 2 });
  const [first, tail] = await Promise.all([gathered.readLease('weights.bin', 0, 4), gathered.readLease('weights.bin', 8, 3)]);
  assert.deepEqual([...first.bytes], [...new Uint8Array(await reference.readRange('weights.bin', 0, 4))]);
  assert.deepEqual([...tail.bytes], [...new Uint8Array(await reference.readRange('weights.bin', 8, 3))]);
  first.release(); tail.release();
  const direct = await gathered.getReceipt();
  assert.equal(direct.directPieceReads, 2);
  assert.equal(direct.directPieceBytes, 7);
  assert.equal(direct.copyMs, 0, 'Exact-piece worker reads perform no explicit verification-copy or gather-copy');
  assert.equal(direct.requestedBytes, 7);
  assert.equal(direct.verifiedBytes, 7);

  for (const [offset, length] of [[2, 4], [0, 8]]) {
    const spanning = await gathered.readLease('weights.bin', offset, length);
    assert.deepEqual([...spanning.bytes], [...new Uint8Array(await reference.readRange('weights.bin', offset, length))]);
    spanning.release();
  }
  const fallback = await gathered.getReceipt();
  assert.equal(fallback.directPieceReads, 2, 'Non-exact requests retain the existing authenticated gathering path');
  assert.equal(fallback.requestedBytes, 19);
  assert.equal(fallback.verifiedBytes, 11);
  const alias = await gathered.readLease('other.bin', 0, 4);
  assert.deepEqual([...alias.bytes], [...bytes.subarray(0, 4)]); alias.release();
  const shared = await gathered.getReceipt();
  assert.equal(shared.verifiedBytes, 11, 'Receipt keeps content deduplication across independently pinned file ranges');
  assert.equal(shared.ranges.length, 4);

  const corrupt = bytes.slice(); corrupt[3] ^= 1;
  await writeFile(join(directory, 'weights.bin'), corrupt);
  await assert.rejects(gathered.readLease('weights.bin', 0, 4), /integrity/);
  await assert.rejects(gathered.readLease('weights.bin', 0, 1), /integrity/, 'Fallback verifies corrupt bytes outside the requested prefix');
  await writeFile(join(directory, 'weights.bin'), bytes.subarray(0, 2));
  await assert.rejects(gathered.readLease('weights.bin', 0, 4), /Incomplete/);
  await writeFile(join(directory, 'weights.bin'), bytes);
  const recovered = await gathered.readLease('weights.bin', 0, 4);
  assert.deepEqual([...recovered.bytes], [...bytes.subarray(0, 4)]); recovered.release();
  assert.equal(recovered.bytes.byteLength, 0, 'Returned output is detached before the worker may overwrite it');
  assert.equal((await gathered.getReceipt()).directPieceReads, 4, 'Failed reads cannot create successful authentication evidence');
} finally { await gathered?.close(); await reference?.close(); await rm(directory, { recursive: true }); }
console.log('direct authenticated piece: generic byte parity, no explicit gather copy, complete SHA, immutable metadata, fallback, failure and recovery passed');
