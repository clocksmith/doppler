import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { buildMerkleTree } from '../../src/formats/rdrr/merkle.js';
import { clearManifest, parseManifest } from '../../src/formats/rdrr/parsing.js';
import {
  cleanup,
  loadShardRange,
  openModelStore,
  saveAuxFile,
  saveManifest,
  verifyIntegrity,
} from '../../src/storage/shard-manager.js';

const canonicalManifest = JSON.parse(
  readFileSync(
    new URL('../../models/local/gemma-3-1b-it-q4k-ehf16-af32/manifest.json', import.meta.url),
    'utf8'
  )
);

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function createTestManifest(modelId, bytes, root) {
  const manifest = clone(canonicalManifest);
  manifest.modelId = modelId;
  manifest.shards = [{
    index: 0,
    filename: 'shard_00000.bin',
    size: bytes.byteLength,
    hash: 'a'.repeat(64),
    offset: 0,
  }];
  manifest.totalSize = bytes.byteLength;
  manifest.tensors = {
    alpha: {
      shard: 0,
      offset: 0,
      size: bytes.byteLength,
      shape: [2, bytes.byteLength / 2],
      dtype: 'f16',
      role: 'weight',
    },
  };
  manifest.integrityExtensions = {
    contractVersion: 1,
    blockMerkle: {
      blockSize: 4,
      roots: {
        alpha: root,
      },
    },
  };
  manifest.inference = {
    ...manifest.inference,
    normalization: {
      ...manifest.inference.normalization,
      postAttentionNorm: false,
      preFeedforwardNorm: false,
      postFeedforwardNorm: false,
    },
  };
  return manifest;
}

async function prepareStoredModel(modelId, manifest, shardBytes) {
  parseManifest(JSON.stringify(manifest));
  await openModelStore(modelId);
  await saveManifest(JSON.stringify(manifest));
  await saveAuxFile('shard_00000.bin', shardBytes);
}

{
  await cleanup();
  clearManifest();
  const shardBytes = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const manifest = createTestManifest(
    'tensor-range-integrity-ok',
    shardBytes,
    buildMerkleTree(shardBytes, { blockSize: 4 }).root
  );
  await prepareStoredModel(manifest.modelId, manifest, shardBytes);
  const range = new Uint8Array(await loadShardRange(0, 2, 4, {
    verify: true,
    tensorId: 'alpha',
  }));
  assert.deepEqual(Array.from(range), [2, 3, 4, 5]);
}

{
  await cleanup();
  clearManifest();
  const shardBytes = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2]);
  const manifest = createTestManifest(
    'tensor-range-integrity-stale',
    shardBytes,
    buildMerkleTree(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]), { blockSize: 4 }).root
  );
  await prepareStoredModel(manifest.modelId, manifest, shardBytes);
  await assert.rejects(
    () => loadShardRange(0, 0, 4, {
      verify: true,
      tensorId: 'alpha',
    }),
    /Tensor integrity mismatch/,
  );
  const integrity = await verifyIntegrity({ checkHashes: false, checkTensorRoots: true });
  assert.equal(integrity.valid, false);
  assert.deepEqual(integrity.corruptTensors, ['alpha']);
}

{
  await cleanup();
  clearManifest();
  const shardBytes = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const manifest = createTestManifest(
    'tensor-range-integrity-cache-invalidates',
    shardBytes,
    buildMerkleTree(shardBytes, { blockSize: 4 }).root
  );
  await prepareStoredModel(manifest.modelId, manifest, shardBytes);
  await loadShardRange(0, 0, 4, {
    verify: true,
    tensorId: 'alpha',
  });
  await saveAuxFile('shard_00000.bin', new Uint8Array([7, 6, 5, 4, 3, 2, 1, 0]));
  await assert.rejects(
    () => loadShardRange(0, 0, 4, {
      verify: true,
      tensorId: 'alpha',
    }),
    /Tensor integrity mismatch/,
  );
}

console.log('shard-manager-tensor-range-integrity.test: ok');
