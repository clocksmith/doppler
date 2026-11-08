import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

import { createNodeFileShardStorageContext } from '../../src/inference/pipelines/text/init.js';

const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-node-file-shards-'));

try {
  const shard0Path = path.join(tempRoot, 'shard-0.bin');
  const shard1Path = path.join(tempRoot, 'shard-1.bin');
  const directSourcePath = path.join(tempRoot, 'model.safetensors');
  const tokenizerJsonPath = path.join(tempRoot, 'tokenizer.json');
  await fs.writeFile(shard0Path, new Uint8Array([0, 1, 2, 3, 4, 5]));
  await fs.writeFile(shard1Path, new Uint8Array([10, 11, 12, 13, 14, 15, 16, 17]));
  await fs.writeFile(directSourcePath, new Uint8Array([21, 22, 23, 24, 25, 26]));
  await fs.writeFile(tokenizerJsonPath, JSON.stringify({ version: '1.0', model: { type: 'BPE' } }));

  const manifest = {
    modelId: 'node-file-shard-storage-test',
    shards: [
      { filename: 'shard-0.bin', size: 6 },
      { filename: 'shard-1.bin', size: 8 },
    ],
  };
  const baseUrl = pathToFileURL(tempRoot).href;
  // This fixture carries no digests, so verification is switched off explicitly.
  const storageContext = createNodeFileShardStorageContext(baseUrl, manifest, { verifyHashes: false });
  assert.equal(storageContext.verifyHashes, false);

  assert.ok(storageContext, 'Expected node file shard storage context for file:// base URL');

  const shard0 = new Uint8Array(await storageContext.loadShard(0));
  assert.deepEqual(Array.from(shard0), [0, 1, 2, 3, 4, 5]);

  const shard1Slice = new Uint8Array(await storageContext.loadShardRange(1, 2, 3));
  assert.deepEqual(Array.from(shard1Slice), [12, 13, 14]);

  const streamedChunks = [];
  for await (const chunk of storageContext.streamShardRange(1, 1, 5, { chunkBytes: 2 })) {
    streamedChunks.push(...new Uint8Array(chunk));
  }
  assert.deepEqual(streamedChunks, [11, 12, 13, 14, 15]);

  const pathStorageContext = createNodeFileShardStorageContext(tempRoot, manifest, { verifyHashes: false });
  assert.ok(pathStorageContext, 'Expected node file shard storage context for plain filesystem path');
  const pathSlice = new Uint8Array(await pathStorageContext.loadShardRange(0, 3, 2));
  assert.deepEqual(Array.from(pathSlice), [3, 4]);

  // Without an explicit decision the context verifies shard digests. Loads by
  // file path used to skip every digest check because the context hardcoded
  // verifyHashes=false and that value shadowed the runtime loading policy.
  const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
  const hashedManifest = {
    modelId: 'node-file-shard-storage-hashed-test',
    hashAlgorithm: 'sha256',
    shards: [
      { filename: 'shard-0.bin', size: 6, hash: sha256(new Uint8Array([0, 1, 2, 3, 4, 5])) },
      { filename: 'shard-1.bin', size: 8, hash: sha256(new Uint8Array([10, 11, 12, 13, 14, 15, 16, 17])) },
    ],
  };
  const verifyingContext = createNodeFileShardStorageContext(baseUrl, hashedManifest);
  assert.equal(verifyingContext.verifyHashes, true, 'node file contexts verify by default');
  assert.deepEqual(Array.from(new Uint8Array(await verifyingContext.loadShard(0))), [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(Array.from(new Uint8Array(await verifyingContext.loadShardRange(1, 2, 3))), [12, 13, 14]);
  await verifyingContext.close?.();

  const tamperedManifest = {
    ...hashedManifest,
    shards: [{ ...hashedManifest.shards[0], hash: sha256(new Uint8Array([9, 9, 9, 9, 9, 9])) }, hashedManifest.shards[1]],
  };
  const tamperedContext = createNodeFileShardStorageContext(baseUrl, tamperedManifest);
  await assert.rejects(() => tamperedContext.loadShard(0), /hash mismatch/);
  await assert.rejects(() => tamperedContext.loadShardRange(0, 1, 2), /hash mismatch/);
  await tamperedContext.close?.();

  const undigestedContext = createNodeFileShardStorageContext(baseUrl, manifest);
  await assert.rejects(() => undigestedContext.loadShard(0), /missing a hash digest/);
  await undigestedContext.close?.();

  const directSourceManifest = {
    modelId: 'node-file-direct-source-test',
    hashAlgorithm: 'sha256',
    shards: [
      { filename: 'source_00000.bin', size: 6 },
    ],
    metadata: {
      sourceRuntime: {
        mode: 'direct-source',
        schema: 'direct-source/v1',
        schemaVersion: 1,
        sourceKind: 'safetensors',
        hashAlgorithm: 'sha256',
        pathSemantics: 'artifact-relative',
        sourceFiles: [
          { index: 0, path: 'model.safetensors', size: 6 },
        ],
        auxiliaryFiles: [
          { path: 'tokenizer.json', size: 39, kind: 'tokenizer_json' },
        ],
        tokenizer: {
          jsonPath: 'tokenizer.json',
          configPath: null,
          modelPath: null,
        },
      },
    },
  };
  const directSourceStorageContext = createNodeFileShardStorageContext(baseUrl, directSourceManifest, { verifyHashes: false });
  assert.ok(directSourceStorageContext, 'Expected node file storage context for direct-source manifest');
  const directSourceBytes = new Uint8Array(await directSourceStorageContext.loadShard(0));
  assert.deepEqual(Array.from(directSourceBytes), [21, 22, 23, 24, 25, 26]);
  const tokenizerJson = await directSourceStorageContext.loadTokenizerJson();
  assert.equal(tokenizerJson.model.type, 'BPE');
} finally {
  await fs.rm(tempRoot, { recursive: true, force: true });
}

console.log('node-file-shard-storage-context.test: ok');
