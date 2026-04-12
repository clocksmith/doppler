import assert from 'node:assert/strict';

import {
  ARTIFACT_FORMAT_DIRECT_SOURCE,
  ARTIFACT_FORMAT_RDRR,
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

console.log('artifact-storage-context.test: ok');
