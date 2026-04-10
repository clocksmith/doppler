import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import {
  cleanup,
  computeSHA256,
  loadManifestFromStore,
  openModelStore,
  saveAuxFile,
  saveManifest,
  writeShard,
} from '../../src/storage/shard-manager.js';
import { setManifest } from '../../src/formats/rdrr/index.js';
import {
  createStoredSourceArtifactContext,
  synthesizeStoredSourceArtifactManifest,
} from '../../src/storage/source-artifact-store.js';

function createManifest(modelId) {
  return {
    version: 1,
    modelId,
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256',
    totalSize: 4,
    architecture: {
      numLayers: 1,
      hiddenSize: 64,
      intermediateSize: 256,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 64,
      vocabSize: 32,
      maxSeqLen: 128,
    },
    inference: {
      layerPattern: {
        type: 'repeat',
        globalPattern: ['attention', 'ffn'],
      },
      session: {
        compute: {
          defaults: {
            activationDtype: 'f16',
            mathDtype: 'f16',
            accumDtype: 'f32',
            outputDtype: 'f16',
          },
        },
        kvcache: {
          layout: 'contiguous',
          kvDtype: 'f16',
          pageSize: 128,
          tiering: {
            mode: 'off',
          },
        },
        decodeLoop: {
          batchSize: 1,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          ringTokens: 1,
          ringStop: 1,
          ringStaging: 1,
          disableCommandBatching: false,
        },
      },
    },
    tensors: {
      'model.embed_tokens.weight': {
        shard: 0,
        offset: 0,
        size: 4,
        shape: [1],
        dtype: 'F32',
        role: 'embedding',
        group: 'embeddings',
      },
    },
    tokenizer: {
      type: 'bundled',
      vocabSize: 1,
      file: 'tokenizer.json',
    },
    eos_token_id: 1,
    shards: [
      {
        index: 0,
        filename: 'model-00001-of-00001.bin',
        size: 4,
        hash: null,
        offset: 0,
      },
    ],
    groups: {
      embeddings: {
        type: 'embedding',
        version: '1.0.0',
        shards: [0],
        tensors: ['model.embed_tokens.weight'],
        hash: '0'.repeat(64),
      },
    },
    metadata: {
      source: 'test',
    },
  };
}

const originalNavigator = globalThis.navigator;

try {
  setRuntimeConfig({
    loading: {
      storage: {
        backend: {
          backend: 'memory',
          memory: {
            maxBytes: 1024 * 1024,
          },
        },
      },
    },
  });

  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        getDirectory() {},
        estimate: async () => ({ usage: 0, quota: 1024 * 1024 }),
        persisted: async () => true,
        persist: async () => true,
      },
    },
    configurable: true,
  });

  const modelId = 'stored-source-artifact-synthesis-contract-model';
  const shardBytes = new Uint8Array([1, 2, 3, 4]);
  const tokenizerJson = { version: '1.0', model: { vocab: [['<bos>', 0]] } };
  const manifest = createManifest(modelId);
  manifest.shards[0].hash = await computeSHA256(shardBytes);

  await openModelStore(modelId);
  setManifest(manifest);
  await saveManifest(JSON.stringify(manifest));
  await writeShard(0, shardBytes, { verify: true });
  await saveAuxFile('tokenizer.json', new TextEncoder().encode(JSON.stringify(tokenizerJson)));

  const storedManifest = JSON.parse(await loadManifestFromStore());
  const synthesized = synthesizeStoredSourceArtifactManifest(storedManifest);
  assert.equal(synthesized.changed, true);
  assert.equal(synthesized.manifest.metadata?.sourceRuntime?.mode, 'direct-source');
  assert.equal(
    synthesized.manifest.metadata?.sourceRuntime?.sourceFiles?.[0]?.path,
    'model-00001-of-00001.bin'
  );
  assert.equal(
    synthesized.manifest.metadata?.sourceRuntime?.tokenizer?.jsonPath,
    'tokenizer.json'
  );

  // Stored shard digests were verified at import time, so warm loads should not
  // rehash the full shard before range reads.
  synthesized.manifest.metadata.sourceRuntime.sourceFiles[0].hash = 'f'.repeat(64);
  const storageContext = createStoredSourceArtifactContext(synthesized.manifest, { verifyHashes: true });
  const shardSlice = await storageContext.loadShardRange(0, 1, 2);
  assert.deepEqual(Array.from(new Uint8Array(shardSlice)), [2, 3]);
  assert.deepEqual(await storageContext.loadTokenizerJson(), tokenizerJson);
} finally {
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
  resetRuntimeConfig();
  await cleanup();
}

console.log('stored-source-artifact-synthesis-contract.test: ok');
