import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createVerifiedPieceReader } from '../../src/storage/verified-piece-reader.js';

const directory = await mkdtemp(join(tmpdir(), 'doppler-piece-index-'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
try {
  const weights = Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]);
  const manifest = { hashAlgorithm: 'sha256',
    shards: [{ index: 0, filename: 'weights.bin', size: weights.length, hash: digest(weights) }],
    tensors: { fixture: { shard: 0, offset: 0, size: weights.length } }, tokenizer: { file: 'tokenizer.json' } };
  await writeFile(join(directory, 'weights.bin'), weights);
  await writeFile(join(directory, 'tokenizer.json'), '{}');
  const writeManifest = () => writeFile(join(directory, 'manifest.json'), JSON.stringify(manifest));
  const run = (destination, layoutPath) => spawnSync(process.execPath, [
    new URL('../../tools/build-verified-piece-index.js', import.meta.url).pathname,
    directory, destination, '4096', ...(layoutPath ? [join(directory, 'manifest.json'), layoutPath] : []),
  ], { encoding: 'utf8' });
  await writeManifest();
  const destination = join(directory, 'pieces.json');
  const good = run(destination);
  assert.equal(good.status, 0, good.stderr);
  const accepted = await readFile(destination, 'utf8');
  assert.equal(JSON.parse(accepted).files[0].pieces[0].identity, 'sha256:' + digest(weights));
  const layoutPath = join(directory, 'layout.json');
  const declared = { path: 'weights.bin', size: weights.length, boundaries: [0, 3, 5, 8] };
  const layout = { schema: 'doppler.verified-piece-layout/v1', files: [declared] };
  await writeFile(layoutPath, JSON.stringify(layout));
  const alignedPath = join(directory, 'aligned.json');
  const aligned = run(alignedPath, layoutPath);
  assert.equal(aligned.status, 0, aligned.stderr);
  const alignedBytes = await readFile(alignedPath);
  const alignedIndex = JSON.parse(alignedBytes);
  assert.deepEqual(alignedIndex.files[0].pieces.map(piece => piece.size), [3, 2, 3]);
  let acquired = 0;
  const reader = await createVerifiedPieceReader({
    manifestBytes: await readFile(join(directory, 'manifest.json')), indexBytes: alignedBytes,
    indexIdentity: 'sha256:' + digest(alignedBytes),
    acquire: async piece => {
      acquired += piece.size;
      return weights.subarray(piece.offset, piece.offset + piece.size);
    },
  });
  try {
    assert.deepEqual(new Uint8Array(await reader.readRange('weights.bin', 3, 2)), new Uint8Array([4, 5]));
    assert.equal(acquired, 2, 'Aligned range authenticates its own piece without unrelated bytes');
  } finally { await reader.close(); }
  for (const invalid of [
    { ...declared, boundaries: [0, 3, 3, 8] },
    { ...declared, boundaries: [0, 5, 4, 8] },
    { ...declared, boundaries: [0, 9, 8] },
    { ...declared, boundaries: [1, 8] },
    { ...declared, boundaries: [0, 7] },
    { ...declared, boundaries: [0, 3.5, 8] },
    { ...declared, size: 9 },
    { ...declared, path: 'unlisted.bin' },
  ]) {
    await writeFile(layoutPath, JSON.stringify({ ...layout, files: [invalid] }));
    const rejected = run(alignedPath, layoutPath);
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /Invalid verified piece layout/);
    assert.deepEqual(await readFile(alignedPath), alignedBytes);
  }
  await writeFile(layoutPath, JSON.stringify({ ...layout, files: [declared, declared] }));
  assert.match(run(alignedPath, layoutPath).stderr, /Invalid verified piece layout/);
  await writeFile(layoutPath, JSON.stringify(layout));
  manifest.shards[0].hash = '0'.repeat(64);
  await writeManifest();
  const bad = run(destination);
  assert.notEqual(bad.status, 0);
  assert.match(bad.stderr, /Source shard hash mismatch/);
  assert.equal(await readFile(destination, 'utf8'), accepted, 'failed validation cannot overwrite an accepted index');
  const badAligned = run(alignedPath, layoutPath);
  assert.notEqual(badAligned.status, 0);
  assert.match(badAligned.stderr, /Source shard hash mismatch/);
  assert.deepEqual(await readFile(alignedPath), alignedBytes, 'Alignment cannot waive whole-source integrity');
  const missing = join(directory, 'rejected.json');
  assert.notEqual(run(missing).status, 0);
  await assert.rejects(access(missing), { code: 'ENOENT' });
  console.log('piece-index producer retains whole-shard integrity before emitting piece identities');
} finally { await rm(directory, { recursive: true, force: true }); }
