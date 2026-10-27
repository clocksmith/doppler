import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { migrateRdrrShardHashes } from '../../tools/migrate-rdrr-shard-hashes.js';
import { createRDRRManifestFixture } from '../helpers/rdrr-manifest-fixture.js';

const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const root = await mkdtemp(path.join(os.tmpdir(), 'doppler-shard-migration-'));
try {
  const sourceDir = path.join(root, 'original');
  await mkdir(sourceDir);
  const bytes = Buffer.from(Array.from({ length: 16 }, (_, index) => index));
  const manifest = createRDRRManifestFixture();
  manifest.hashAlgorithm = 'blake3';
  manifest.shards[0].blake3 = manifest.shards[0].hash;
  manifest.artifactIdentity = {
    sourceCheckpointId: 'synthetic/source', sourceFormat: 'safetensors',
    manifestVariantId: 'old-variant', weightPackId: 'old-weights',
    weightPackHash: `sha256:${'1'.repeat(64)}`, shardSetHash: `sha256:${'2'.repeat(64)}`,
  };
  manifest.tensors = {
    weight: { shard: 0, offset: 0, size: 16, dtype: 'F16', shape: [2, 4], role: 'embedding' },
  };
  manifest.integrityExtensions = {
    contractVersion: 1, blockMerkle: { blockSize: 8, roots: { weight: `sha256:${'0'.repeat(64)}` } },
  };
  const original = `${JSON.stringify(manifest)}\n`;
  await writeFile(path.join(sourceDir, 'manifest.json'), original);
  await writeFile(path.join(sourceDir, 'shard_00000.bin'), bytes);
  const inventory = JSON.stringify([
    { path: 'models/synthetic/manifest.json', size: Buffer.byteLength(original),
      oid: createHash('sha1').update(`blob ${Buffer.byteLength(original)}\0`).update(original).digest('hex') },
    { path: 'models/synthetic/shard_00000.bin', size: bytes.length,
      lfs: { oid: sha256(bytes) } },
  ]);
  const inventoryPath = path.join(root, 'independent-inventory.json');
  await writeFile(inventoryPath, inventory);
  const options = { sourceDir, destinationDir: path.join(root, 'migrated'),
    inventoryPath, inventoryHash: sha256(inventory), inventoryPrefix: 'models/synthetic' };
  await assert.rejects(migrateRdrrShardHashes({ ...options, destinationDir: sourceDir }), /separate destination/);
  await assert.rejects(migrateRdrrShardHashes({ ...options, inventoryHash: '0'.repeat(64) }), /inventory SHA-256 mismatch/);
  await writeFile(path.join(sourceDir, 'shard_00000.bin'), Buffer.alloc(16));
  await assert.rejects(migrateRdrrShardHashes(options), /inventory byte mismatch/);
  await writeFile(path.join(sourceDir, 'shard_00000.bin'), bytes);
  const first = await migrateRdrrShardHashes(options);
  assert.equal(first.manifest.hashAlgorithm, 'sha256');
  assert.equal(first.manifest.shards[0].hash, sha256(bytes));
  assert.equal(first.manifest.shards[0].sha256, sha256(bytes));
  assert.equal(first.manifest.shards[0].blake3, undefined);
  assert.notEqual(first.manifest.artifactIdentity.manifestVariantId, 'old-variant');
  assert.notEqual(first.manifest.artifactIdentity.shardSetHash, manifest.artifactIdentity.shardSetHash);
  assert.equal(first.manifest.artifactIdentity.weightPackId, undefined);
  assert.match(first.manifest.artifactIdentity.weightCapsuleHash, /^sha256:[0-9a-f]{64}$/);
  assert.deepEqual(first.manifest.tensors, manifest.tensors);
  assert.deepEqual(first.manifest.inference, manifest.inference);
  assert.notEqual(first.manifest.integrityExtensions.blockMerkle.roots.weight,
    manifest.integrityExtensions.blockMerkle.roots.weight);
  assert.equal(first.receipt.qualificationTransferred, false);
  assert.equal(await readFile(path.join(sourceDir, 'manifest.json'), 'utf8'), original);
  const second = await migrateRdrrShardHashes({ ...options, destinationDir: path.join(root, 'repeated') });
  assert.equal(first.receipt.manifestHash, second.receipt.manifestHash, 'migration identities must reproduce');
  await assert.rejects(migrateRdrrShardHashes(options), /EEXIST/);
} finally {
  await rm(root, { recursive: true, force: true });
}
console.log('rdrr-shard-hash-migration.test: ok');
