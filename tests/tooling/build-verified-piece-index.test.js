import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

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
  const run = destination => spawnSync(process.execPath, [
    new URL('../../tools/build-verified-piece-index.js', import.meta.url).pathname,
    directory, destination, '4096',
  ], { encoding: 'utf8' });
  await writeManifest();
  const destination = join(directory, 'pieces.json');
  const good = run(destination);
  assert.equal(good.status, 0, good.stderr);
  const accepted = await readFile(destination, 'utf8');
  assert.equal(JSON.parse(accepted).files[0].pieces[0].identity, 'sha256:' + digest(weights));
  manifest.shards[0].hash = '0'.repeat(64);
  await writeManifest();
  const bad = run(destination);
  assert.notEqual(bad.status, 0);
  assert.match(bad.stderr, /Source shard hash mismatch/);
  assert.equal(await readFile(destination, 'utf8'), accepted, 'failed validation cannot overwrite an accepted index');
  const missing = join(directory, 'rejected.json');
  assert.notEqual(run(missing).status, 0);
  await assert.rejects(access(missing), { code: 'ENOENT' });
  console.log('piece-index producer retains whole-shard integrity before emitting piece identities');
} finally { await rm(directory, { recursive: true, force: true }); }
