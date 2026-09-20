import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createCapsuleArtifactSource } from '../../src/client/runtime/capsule-artifact-source.js';
import { createVerifiedCapsuleArtifactStore, isVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { computeBlake3 } from '../../src/storage/shards/integrity.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function fixture({ size = 8 * 1024 * 1024, shard = {} } = {}) {
  const weights = new Uint8Array(size).fill(23);
  const manifest = { modelId: 'verification-reuse', hashAlgorithm: 'sha256',
    shards: [{ filename: 'weights.bin', size, hash: hash(weights), ...shard }] };
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const artifacts = [
    { artifactId: 'manifest', role: 'manifest', path: 'model/manifest.json', sizeBytes: manifestBytes.length, hash: `sha256:${hash(manifestBytes)}` },
    { artifactId: 'weights', role: 'weight-shard', path: 'model/weights.bin', sizeBytes: size, hash: `sha256:${hash(weights)}` },
  ];
  const capsule = { modelId: manifest.modelId, program: { manifestArtifactId: 'manifest' }, artifacts };
  let sourceReads = 0;
  const source = { async readArtifact(artifact) {
    sourceReads++;
    return artifact.artifactId === 'manifest' ? manifestBytes : weights;
  } };
  return { capsule, source, weights, reads: () => sourceReads };
}

// A complete model read consumes one copy, not a verification copy plus a model
// copy. The same owned snapshot authenticates the manifest and every later range.
{
  const f = fixture();
  const store = createVerifiedCapsuleArtifactStore(f.capsule, f.source);
  assert(isVerifiedCapsuleArtifactStore(store));
  assert(Object.isFrozen(store));
  assert.equal(Reflect.set(store, 'readArtifactRange', async () => new Uint8Array(0)), false);
  const source = await createCapsuleArtifactSource(f.capsule, store);
  const before = store.getMetrics();
  f.weights.fill(0);
  const result = new Uint8Array(await source.storageContext.loadShard(0));
  assert(result.every(value => value === 23));
  assert.equal(store.getMetrics().returnedBytes - before.returnedBytes, result.length);
  assert.equal(store.getMetrics().hashedBytes, before.hashedBytes);
  assert.equal(f.reads(), 2, 'later range reads do not reacquire mutable source bytes');
  result.fill(0);
  assert.deepEqual(new Uint8Array(await source.storageContext.loadShardRange(0, 7, 3)), new Uint8Array([23, 23, 23]));
  store.close();
  await assert.rejects(source.storageContext.loadShard(0), /closed/);
}

// Signed descriptors disagreeing about the bytes fail before model execution.
for (const shard of [{ hash: '0'.repeat(64) }, { hash: null }, { size: 9 }, { filename: 'outside.bin' }]) {
  const f = fixture({ size: 8, shard });
  const store = createVerifiedCapsuleArtifactStore(f.capsule, f.source);
  await assert.rejects(createCapsuleArtifactSource(f.capsule, store), /hash or size mismatch|outside/);
  store.close();
}

// Reuse follows the storage owner's canonical path/hash normalization.
{
  const f = fixture({ size: 8, shard: { filename: ' /weights.bin ', hashAlgorithm: ' SHA256 ' } });
  const store = createVerifiedCapsuleArtifactStore(f.capsule, f.source);
  const source = await createCapsuleArtifactSource(f.capsule, store);
  assert.equal((await source.storageContext.loadShard(0)).byteLength, 8);
  store.close();
}

// An advertised digest, copied interface, or proxy does not confer ownership.
{
  const f = fixture({ size: 8 });
  const store = createVerifiedCapsuleArtifactStore(f.capsule, f.source);
  for (const facade of [{ ...store }, new Proxy(store, {})]) {
    assert.equal(isVerifiedCapsuleArtifactStore(facade), false);
    const source = await createCapsuleArtifactSource(f.capsule, facade);
    const before = store.getMetrics().returnedBytes;
    await source.storageContext.loadShard(0);
    assert.equal(store.getMetrics().returnedBytes - before, 16, 'generic ports retain byte verification');
  }
  store.close();
}
{
  const f = fixture({ size: 8 });
  let claimedHashes = 0;
  const forged = { ...f.source, async hashArtifact(artifact) { claimedHashes++; return { hash: artifact.hash, sizeBytes: artifact.sizeBytes }; } };
  const source = await createCapsuleArtifactSource(f.capsule, forged);
  f.weights.fill(0);
  await assert.rejects(source.storageContext.loadShard(0), /hash mismatch/);
  assert.equal(claimedHashes, 0);
}

// A BLAKE3 manifest is not authenticated by the Capsule's SHA-256 digest alone.
{
  const bytes = new Uint8Array(8).fill(23);
  for (const expected of [await computeBlake3(bytes), '0'.repeat(64)]) {
    const f = fixture({ size: 8, shard: { hashAlgorithm: 'blake3', hash: expected } });
    const store = createVerifiedCapsuleArtifactStore(f.capsule, f.source);
    const source = await createCapsuleArtifactSource(f.capsule, store);
    const before = store.getMetrics().returnedBytes;
    if (expected === '0'.repeat(64)) {
      await assert.rejects(source.storageContext.loadShard(0), /hash mismatch/);
    } else {
      assert.deepEqual(new Uint8Array(await source.storageContext.loadShard(0)), bytes);
      assert.equal(store.getMetrics().returnedBytes - before, 16);
    }
    store.close();
  }
}
console.log('capsule-artifact-verification-reuse: passed');
