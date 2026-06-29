import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '../..');
const SHARDS_DIR = path.join(ROOT, 'tools/functional_shards');

async function testFunctionalTensorArtifacts() {
  console.log('Testing functional tensor artifacts...');

  // 1. Verify manifest.json
  const manifestPath = path.join(SHARDS_DIR, 'manifest.json');
  const manifestRaw = await fs.readFile(manifestPath, 'utf8');
  const manifest = JSON.parse(manifestRaw);

  assert.equal(manifest.schema_version, 'manifoldgguf.v0.1');
  assert.equal(manifest.tensor_name, 'model.layers.0.mlp.down_proj');
  assert.equal(manifest.storage_type, 'functional_descriptor');
  assert.deepEqual(manifest.shape, [896, 4864]);
  assert.ok(manifest.descriptor_hash);
  assert.ok(manifest.source_tensor_hash);

  // Verify components
  const comps = manifest.components;
  assert.equal(comps.prng_substrate.algorithm, 'coordinate_hash_normal_v1');
  assert.equal(comps.kronecker_sum.shard_file, 'layers_0_down_proj.kron');
  assert.equal(comps.coordinate_inr.shard_file, 'layers_0_down_proj.siren');
  assert.equal(comps.sparse_outliers.shard_file, 'layers_0_down_proj.sparse');

  // 2. Verify Kronecker factor file
  const kronPath = path.join(SHARDS_DIR, 'layers_0_down_proj.kron');
  const kron = JSON.parse(await fs.readFile(kronPath, 'utf8'));
  assert.ok(Array.isArray(kron.A));
  assert.ok(Array.isArray(kron.B));
  assert.equal(kron.A.length, comps.kronecker_sum.rank_terms);

  // 3. Verify SIREN weights file
  const sirenPath = path.join(SHARDS_DIR, 'layers_0_down_proj.siren');
  const siren = JSON.parse(await fs.readFile(sirenPath, 'utf8'));
  assert.ok(siren['net.0.weight']);

  // 4. Verify sparse outliers file
  const sparsePath = path.join(SHARDS_DIR, 'layers_0_down_proj.sparse');
  const sparse = JSON.parse(await fs.readFile(sparsePath, 'utf8'));
  assert.ok(Array.isArray(sparse.values));
  assert.ok(Array.isArray(sparse.col_indices));
  assert.ok(Array.isArray(sparse.row_offsets));
  assert.equal(sparse.row_offsets.length, manifest.shape[0] + 1);

  // 5. Verify build receipt file
  const receiptPath = path.join(SHARDS_DIR, 'layers_0_down_proj.json.receipt.json');
  const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
  assert.equal(receipt.schema, 'simulatte.indexBuildReceipt.v1');
  assert.ok(Number.isFinite(receipt.byteSize));
  assert.ok(receipt.sha256);

  console.log('✓ functional-tensor artifacts test: ok');
}

testFunctionalTensorArtifacts().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
