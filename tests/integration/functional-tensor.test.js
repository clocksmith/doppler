import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { assertFunctionalDescriptorManifest } from '../../src/formats/rdrr/functional-descriptor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '../..');
const SHARDS_DIR = path.join(ROOT, 'tools/functional_shards');

function sha256(bytes) {
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

function readU32LE(bytes, offset = 0) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true);
}

async function testFunctionalTensorArtifacts() {
  console.log('Testing functional tensor artifacts...');

  // 1. Verify manifest.json
  const manifestPath = path.join(SHARDS_DIR, 'manifest.json');
  const manifestRaw = await fs.readFile(manifestPath, 'utf8');
  const manifest = JSON.parse(manifestRaw);
  assertFunctionalDescriptorManifest(manifest, 'tools/functional_shards/manifest.json');

  assert.equal(manifest.schema_version, 'manifoldgguf.v0.1');
  assert.equal(manifest.tensor_name, 'synthetic.weight');
  assert.equal(manifest.storage_type, 'functional_descriptor');
  assert.deepEqual(manifest.slice_shape, [64, 64]);
  assert.deepEqual(manifest.crop_shape, [64, 64]);
  assert.deepEqual(manifest.padded_shape, [64, 64]);
  assert.equal(manifest.dense_f16_bytes, 8192);
  assert.equal(manifest.descriptor_bytes, 672);
  assert.equal(manifest.proof_status, 'passed');
  assert.deepEqual(manifest.proof_status_gate, {
    sensitivity: 'passed',
    compression: 'passed',
    determinism: 'passed',
  });
  assert.ok(manifest.descriptor_hash);
  assert.ok(manifest.source_tensor_hash);

  // Verify components
  const comps = manifest.components;
  assert.equal(comps.prng_substrate.algorithm, 'coord_hash_normal_v1');
  assert.equal(typeof comps.prng_substrate.learned_scale, 'number');
  assert.equal(comps.kronecker_sum.shard_file, 'synthetic_weight.kron');
  assert.equal(comps.coordinate_inr.shard_file, 'synthetic_weight.siren');
  assert.equal(comps.sparse_outliers.format, 'coo_v1');
  assert.equal(comps.sparse_outliers.shard_file, 'synthetic_weight.sparse');

  // 2. Verify Kronecker factor file
  const kronPath = path.join(SHARDS_DIR, comps.kronecker_sum.shard_file);
  const kron = await fs.readFile(kronPath);
  assert.equal(sha256(kron), comps.kronecker_sum.shard_hash);
  assert.equal(readU32LE(kron), comps.kronecker_sum.rank_terms);

  // 3. Verify SIREN weights file
  const sirenPath = path.join(SHARDS_DIR, comps.coordinate_inr.shard_file);
  const siren = await fs.readFile(sirenPath);
  assert.equal(sha256(siren), comps.coordinate_inr.shard_hash);
  assert.equal(readU32LE(siren), comps.coordinate_inr.network_dims.length - 1);

  // 4. Verify sparse outliers file
  const sparsePath = path.join(SHARDS_DIR, comps.sparse_outliers.shard_file);
  const sparse = await fs.readFile(sparsePath);
  assert.equal(sha256(sparse), comps.sparse_outliers.shard_hash);
  assert.equal(readU32LE(sparse), comps.sparse_outliers.actual_nnz);
  assert.equal(kron.byteLength + siren.byteLength + sparse.byteLength, manifest.descriptor_bytes);
  assert.equal(sha256(Buffer.concat([kron, siren, sparse])), manifest.descriptor_hash);

  // 5. Verify hash and metrics sidecars
  const hashes = JSON.parse(await fs.readFile(path.join(SHARDS_DIR, 'hashes.json'), 'utf8'));
  assert.equal(hashes.descriptor, manifest.descriptor_hash);
  const metrics = JSON.parse(await fs.readFile(path.join(SHARDS_DIR, 'metrics.json'), 'utf8'));
  assert.equal(metrics.dense_f16_bytes, manifest.dense_f16_bytes);
  assert.equal(metrics.descriptor_bytes, manifest.descriptor_bytes);
  assert.equal(metrics.proof_status, manifest.proof_status);

  console.log('functional-tensor artifacts test: ok');
}

testFunctionalTensorArtifacts().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
