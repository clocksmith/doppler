import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';

import {
  ARTIFACT_FORMAT_DIRECT_SOURCE,
  ARTIFACT_FORMAT_RDRR,
  createNodeFileArtifactStorageContext,
  getArtifactFormat,
} from '../../src/storage/artifact-storage-context.js';

{
  const directSourceManifest = {
    modelId: 'direct-source-format-test',
    shards: [{ filename: 'source_00000.bin', size: 16 }],
    metadata: {
      sourceRuntime: {
        mode: 'direct-source',
        sourceKind: 'safetensors',
      },
    },
  };
  assert.equal(getArtifactFormat(directSourceManifest), ARTIFACT_FORMAT_DIRECT_SOURCE);
}

{
  const rdrrManifest = {
    modelId: 'rdrr-format-test',
    shards: [{ filename: 'shard_00000.bin', size: 16 }],
  };
  assert.equal(getArtifactFormat(rdrrManifest), ARTIFACT_FORMAT_RDRR);
}

{
  const storedRdrrManifest = {
    modelId: 'stored-rdrr-format-test',
    shards: [{ filename: 'shard_00000.bin', size: 16 }],
    metadata: {
      sourceRuntime: {
        mode: 'direct-source',
        sourceKind: 'rdrr',
      },
    },
  };
  assert.equal(getArtifactFormat(storedRdrrManifest), ARTIFACT_FORMAT_RDRR);
}

{
  assert.equal(getArtifactFormat({ modelId: 'invalid-format-test' }), null);
}

{
  const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-artifact-storage-context-'));
  const shardPath = path.join(fixtureDir, 'source_00000.bin');
  const manifest = {
    modelId: 'artifact-node-file-reader-cache-test',
    shards: [{ filename: 'source_00000.bin', size: 32 }],
  };
  writeFileSync(shardPath, Uint8Array.from({ length: 32 }, (_, index) => index));
  let storageContext = null;
  try {
    storageContext = createNodeFileArtifactStorageContext(fixtureDir, manifest);
    assert.ok(storageContext, 'node file artifact storage context should be created for filesystem manifests');
    assert.equal(typeof storageContext.close, 'function');
    const firstRange = await storageContext.loadShardRange(0, 0, 8);
    const secondRange = await storageContext.loadShardRange(0, 8, 8);
    assert.equal(new Uint8Array(firstRange).byteLength, 8);
    assert.equal(new Uint8Array(secondRange).byteLength, 8);
    await storageContext.close?.();
  } finally {
    await storageContext?.close?.();
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

console.log('artifact-storage-context.test: ok');
