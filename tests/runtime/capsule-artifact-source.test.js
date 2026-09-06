import assert from 'node:assert/strict';
import { createCapsuleArtifactSource } from '../../src/client/runtime/capsule-artifact-source.js';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';
import { getStorageShaderSourceScope, runWithShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { loadShaderSource } from '../../src/gpu/kernels/shader-cache.js';

const weights = new Uint8Array([1, 2, 3, 4]);
const tokenizer = new TextEncoder().encode('{"type":"test"}');
function fixture(tokenizerPath = 'tokenizer.json') {
  const manifest = { modelId: 'source-test', hashAlgorithm: 'sha256', shards: [{ filename: 'weights.bin', size: 4, hash: hashBytesSha256(weights).slice(7) }], tokenizer: { type: 'bundled', file: tokenizerPath } };
  const bytes = new Map([
    ['manifest', new TextEncoder().encode(JSON.stringify(manifest))],
    ['weights', weights], ['tokenizer', tokenizer],
  ]);
  const artifacts = [['manifest', 'manifest.json'], ['weights', 'weights.bin'], ['tokenizer', 'tokenizer.json']].map(([artifactId, path]) => ({ artifactId, path: `model/${path}`, hash: hashBytesSha256(bytes.get(artifactId)), sizeBytes: bytes.get(artifactId).byteLength }));
  const capsule = { modelId: 'source-test', program: { manifestArtifactId: 'manifest' }, artifacts };
  let reads = 0;
  const store = createVerifiedCapsuleArtifactStore(capsule, { async readArtifact(artifact) { reads += 1; return bytes.get(artifact.artifactId); } });
  return { capsule, store, reads: () => reads };
}
const originalFetch = globalThis.fetch;
{
  const bytes = Buffer.from([1, 2, 3]);
  const artifact = { artifactId: 'node-buffer', path: 'bytes', hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
  const store = createVerifiedCapsuleArtifactStore({ artifacts: [artifact] }, { readArtifact: async () => bytes });
  await store.readArtifact(artifact);
  bytes.fill(0);
  assert.deepEqual(await store.readArtifact(artifact), new Uint8Array([1, 2, 3]));
  store.close();
}
{
  const bytes = Buffer.alloc(1024 * 1024, 7);
  const artifact = { artifactId: 'range', path: 'weights', hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
  const store = createVerifiedCapsuleArtifactStore({ artifacts: [artifact] }, { readArtifact: async () => bytes });
  await store.hashArtifact(artifact);
  assert.equal(store.getMetrics().copiedBytes, bytes.byteLength, 'hash receipt does not copy verified bytes out');
  for (let index = 0; index < 100; index += 1) {
    const range = await store.readArtifactRange(artifact, index, 4);
    assert.equal(range.byteLength, 4);
    assert.equal(range.buffer.byteLength, 4);
    assert.deepEqual(range, new Uint8Array([7, 7, 7, 7]));
    range.fill(0);
  }
  assert.equal(store.getMetrics().hashedBytes, bytes.byteLength, 'shared content is hashed once');
  assert.equal(store.getMetrics().copiedBytes, bytes.byteLength + 400, 'ranges copy only requested bytes');
  bytes.fill(0);
  assert.deepEqual(await store.readArtifactRange(artifact, 0, 4), new Uint8Array([7, 7, 7, 7]));
  for (const [offset, length] of [[-1, 1], [0, -1], [0.5, 1], [0, NaN], [bytes.byteLength, 1], [Number.MAX_SAFE_INTEGER, 1]]) {
    await assert.rejects(store.readArtifactRange(artifact, offset, length), /bounds/);
  }
  await assert.rejects(store.readArtifactRange({ ...artifact, hash: `sha256:${'0'.repeat(64)}` }, 0, 1), /closure/);
  store.close();
  assert.equal(store.getMetrics().retainedBytes, 0);
  await assert.rejects(store.readArtifactRange(artifact, 0, 1), /closed/);
}
globalThis.fetch = async () => { throw new Error('Capsule loading must not refetch from an origin'); };
try {
  const f = fixture();
  const shaderBytes = new TextEncoder().encode('verified Capsule shader');
  const shader = { artifactId: 'shader', role: 'wgsl-source', path: 'shader.wgsl', hash: hashBytesSha256(shaderBytes), sizeBytes: shaderBytes.length };
  const shaderFixture = fixture();
  shaderFixture.capsule.artifacts.push(shader);
  shaderFixture.capsule.wgslModules = [{ id: 'shader', file: 'example.wgsl', sourceArtifactId: shader.artifactId }];
  const shaderStore = createVerifiedCapsuleArtifactStore(shaderFixture.capsule, { readArtifact: async (entry) =>
    entry.artifactId === 'shader' ? shaderBytes : shaderFixture.store.readArtifact(entry) });
  const shaderSource = await createCapsuleArtifactSource(shaderFixture.capsule, shaderStore);
  await runWithShaderSourceScope(getStorageShaderSourceScope(shaderSource.storageContext), async () => {
    assert.equal(await loadShaderSource('example.wgsl'), 'verified Capsule shader');
    await assert.rejects(loadShaderSource('outside.wgsl'), /outside.*closure/);
  });
  shaderStore.close();
  shaderFixture.store.close();
  const source = await createCapsuleArtifactSource(f.capsule, f.store);
  assert.deepEqual(new Uint8Array(await source.storageContext.loadShardRange(0, 1, 2)), new Uint8Array([2, 3]));
  assert.deepEqual(await source.storageContext.loadTokenizerJson(), { type: 'test' });
  await source.storageContext.loadShardRange(0, 0, 4);
  assert.equal(f.reads(), 3);
  assert.deepEqual(new Uint8Array(await source.storageContext.loadShardRange(0, 3, 2)), new Uint8Array([4]));
  f.store.close();
  await assert.rejects(source.storageContext.loadShardRange(0, 0, 1), /closed/);
  const outside = fixture('https://other.invalid/tokenizer.json');
  const bad = await createCapsuleArtifactSource(outside.capsule, outside.store);
  await assert.rejects(bad.storageContext.loadTokenizerJson(), /outside/);
  outside.store.close();
  const alias = fixture();
  alias.capsule.artifacts.push({ ...alias.capsule.artifacts[0], artifactId: 'alias', path: 'model/nested/../manifest.json' });
  await assert.rejects(createCapsuleArtifactSource(alias.capsule, alias.store), /alias/);
  alias.store.close();
} finally { globalThis.fetch = originalFetch; }
console.log('✔ capsule-artifact-source.test.js passed');
