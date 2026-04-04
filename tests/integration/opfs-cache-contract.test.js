import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { cleanup, loadManifestFromStore, openModelStore, saveManifest } from '../../src/storage/shard-manager.js';
import { ensureModelCached } from '../../src/tooling/opfs-cache.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const originalNavigator = globalThis.navigator;
const originalFetch = globalThis.fetch;

function createManifest(modelId) {
  return {
    version: 1,
    modelId,
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256',
    totalSize: 1,
    architecture: {
      numLayers: 1,
      hiddenSize: 64,
      intermediateSize: 256,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 64,
      vocabSize: 32000,
      maxSeqLen: 1024,
    },
    inference: {
      ...clone(DEFAULT_MANIFEST_INFERENCE),
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
          pageSize: 256,
          tiering: {
            mode: 'off',
          },
        },
        decodeLoop: {
          batchSize: 4,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          ringTokens: 1,
          ringStop: 1,
          ringStaging: 1,
          disableCommandBatching: false,
        },
      },
    },
    eos_token_id: 1,
    shards: [
      {
        index: 0,
        filename: 'model-00001-of-00001.bin',
        size: 1,
        hash: '0'.repeat(64),
        offset: 0,
      },
    ],
    groups: {
      layers: {
        type: 'layer',
        version: '1.0.0',
        shards: [0],
        tensors: [],
        hash: '0'.repeat(64),
      },
    },
  };
}

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
      },
    },
    configurable: true,
  });

  await openModelStore('opfs-cache-contract-model');
  await saveManifest(JSON.stringify(createManifest('opfs-cache-contract-model')));

  globalThis.fetch = async () => {
    throw new Error('network unavailable');
  };

  const failed = await ensureModelCached(
    'opfs-cache-contract-model',
    'https://example.test/model'
  );
  assert.equal(failed.cached, false);
  assert.equal(failed.fromCache, false);
  assert.match(String(failed.error || ''), /network unavailable/);

  const cachedManifest = createManifest('opfs-cache-contract-model');
  const remoteManifest = clone(cachedManifest);
  remoteManifest.inference.attention.valueNorm = true;
  remoteManifest.inference.ffn.useDoubleWideMlp = true;
  const remoteManifestText = JSON.stringify(remoteManifest);

  await saveManifest(JSON.stringify(cachedManifest));
  globalThis.fetch = async () => new Response(remoteManifestText, { status: 200 });

  const refreshed = await ensureModelCached(
    'opfs-cache-contract-model',
    'https://example.test/model'
  );
  assert.equal(refreshed.cached, true);
  assert.equal(refreshed.fromCache, false);
  assert.equal(refreshed.cacheState, 'manifest-refresh');

  const storedManifestText = await loadManifestFromStore();
  assert.equal(storedManifestText, remoteManifestText);
} finally {
  globalThis.fetch = originalFetch;
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

console.log('opfs-cache-contract.test: ok');
