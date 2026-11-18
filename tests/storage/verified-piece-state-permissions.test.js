import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, chmod, readdir, writeFile, symlink, unlink, rm } from 'node:fs/promises';
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


test('persisted authentication rejects a group-writable state directory', async () => {
 const root = await mkdtemp(join(tmpdir(), 'state-mode-'));
 let source;
 try {
  const pieces = join(root, 'pieces'), state = join(root, 'state');
  await mkdir(pieces); await mkdir(state); await chmod(state, 0o770);
  await writeFile(join(pieces, 'weights.bin.0'), good);
  await assert.rejects(async () => {
   source = await createVerifiedPieceStorage({...metadata, nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: {directory: state} }});
  }, /permission|writ|private|mode|owned/);
 } finally { await source?.storage.close(); await rm(root, {recursive:true,force:true}); }
});

test('persisted authentication does not trust a group-writable state file', async () => {
 const root = await mkdtemp(join(tmpdir(), 'state-file-mode-'));
 let source;
 try {
  const pieces = join(root, 'pieces'), state = join(root, 'state');
  await mkdir(pieces); await writeFile(join(pieces, 'weights.bin.0'), good);
  const options = {...metadata, nodePieceDirectory: { directory: pieces, maxPieceBytes: 64, maxActiveReadBytes: 64, verificationCache: {directory: state} }};
  source = await createVerifiedPieceStorage(options);
  await source.storage.loadShardRange(0,0,64); await source.storage.close(); source=null;
  for (const file of await readdir(state)) await chmod(join(state,file),0o660);
  source = await createVerifiedPieceStorage(options);
  await source.storage.loadShardRange(0,0,64);
  assert.equal(source.getReceipt().stateCacheHits ?? 0,0);
 } finally { await source?.storage.close(); await rm(root, {recursive:true,force:true}); }
});
