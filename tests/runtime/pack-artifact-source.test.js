import assert from 'node:assert/strict';
import { createPackArtifactSource } from '../../src/client/runtime/pack-artifact-source.js';
import { createVerifiedPackArtifactStore } from '../../src/client/runtime/verified-pack-artifact-store.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

const weights = new Uint8Array([1, 2, 3, 4]);
const tokenizer = new TextEncoder().encode('{"type":"test"}');
function fixture(tokenizerPath = 'tokenizer.json') {
  const manifest = { modelId: 'source-test', hashAlgorithm: 'sha256', shards: [{ filename: 'weights.bin', size: 4, hash: hashBytesSha256(weights).slice(7) }], tokenizer: { type: 'bundled', file: tokenizerPath } };
  const bytes = new Map([
    ['manifest', new TextEncoder().encode(JSON.stringify(manifest))],
    ['weights', weights], ['tokenizer', tokenizer],
  ]);
  const artifacts = [['manifest', 'manifest.json'], ['weights', 'weights.bin'], ['tokenizer', 'tokenizer.json']].map(([artifactId, path]) => ({ artifactId, path: `model/${path}`, hash: hashBytesSha256(bytes.get(artifactId)), sizeBytes: bytes.get(artifactId).byteLength }));
  const pack = { modelId: 'source-test', program: { manifestArtifactId: 'manifest' }, artifacts };
  let reads = 0;
  const store = createVerifiedPackArtifactStore(pack, { async readArtifact(artifact) { reads += 1; return bytes.get(artifact.artifactId); } });
  return { pack, store, reads: () => reads };
}
const originalFetch = globalThis.fetch;
{
  const bytes = Buffer.from([1, 2, 3]);
  const artifact = { artifactId: 'node-buffer', path: 'bytes', hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
  const store = createVerifiedPackArtifactStore({ artifacts: [artifact] }, { readArtifact: async () => bytes });
  await store.readArtifact(artifact);
  bytes.fill(0);
  assert.deepEqual(await store.readArtifact(artifact), new Uint8Array([1, 2, 3]));
  store.close();
}
globalThis.fetch = async () => { throw new Error('Pack loading must not refetch from an origin'); };
try {
  const f = fixture();
  const source = await createPackArtifactSource(f.pack, f.store);
  assert.deepEqual(new Uint8Array(await source.storageContext.loadShardRange(0, 1, 2)), new Uint8Array([2, 3]));
  assert.deepEqual(await source.storageContext.loadTokenizerJson(), { type: 'test' });
  await source.storageContext.loadShardRange(0, 0, 4);
  assert.equal(f.reads(), 3);
  assert.deepEqual(new Uint8Array(await source.storageContext.loadShardRange(0, 3, 2)), new Uint8Array([4]));
  f.store.close();
  await assert.rejects(source.storageContext.loadShardRange(0, 0, 1), /closed/);
  const outside = fixture('https://other.invalid/tokenizer.json');
  const bad = await createPackArtifactSource(outside.pack, outside.store);
  await assert.rejects(bad.storageContext.loadTokenizerJson(), /outside/);
  outside.store.close();
  const alias = fixture();
  alias.pack.artifacts.push({ ...alias.pack.artifacts[0], artifactId: 'alias', path: 'model/nested/../manifest.json' });
  await assert.rejects(createPackArtifactSource(alias.pack, alias.store), /alias/);
  alias.store.close();
} finally { globalThis.fetch = originalFetch; }
console.log('✔ pack-artifact-source.test.js passed');
