import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

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
  const storageContext = createNodeFileShardStorageContext(baseUrl, manifest);

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

  const pathStorageContext = createNodeFileShardStorageContext(tempRoot, manifest);
  assert.ok(pathStorageContext, 'Expected node file shard storage context for plain filesystem path');
  const pathSlice = new Uint8Array(await pathStorageContext.loadShardRange(0, 3, 2));
  assert.deepEqual(Array.from(pathSlice), [3, 4]);

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
  const directSourceStorageContext = createNodeFileShardStorageContext(baseUrl, directSourceManifest);
  assert.ok(directSourceStorageContext, 'Expected node file storage context for direct-source manifest');
  const directSourceBytes = new Uint8Array(await directSourceStorageContext.loadShard(0));
  assert.deepEqual(Array.from(directSourceBytes), [21, 22, 23, 24, 25, 26]);
  const tokenizerJson = await directSourceStorageContext.loadTokenizerJson();
  assert.equal(tokenizerJson.model.type, 'BPE');
} finally {
  await fs.rm(tempRoot, { recursive: true, force: true });
}

console.log('node-file-shard-storage-context.test: ok');
