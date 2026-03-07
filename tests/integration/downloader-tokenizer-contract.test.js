import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { computeHash, cleanup } from '../../src/storage/shard-manager.js';
import { downloadModel } from '../../src/storage/downloader.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const originalNavigator = globalThis.navigator;
const originalFetch = globalThis.fetch;

const shardBytes = new Uint8Array([1, 2, 3, 4]);
const shardHash = await computeHash(shardBytes, 'sha256');

function createManifest() {
  return {
    version: 1,
    modelId: 'manifest-model-id',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256',
    totalSize: shardBytes.byteLength,
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
      presetId: 'gemma3',
    },
    eos_token_id: 1,
    tokenizer: {
      type: 'bundled',
      file: 'tokenizer.json',
    },
    shards: [
      {
        index: 0,
        filename: 'model-00001-of-00001.bin',
        size: shardBytes.byteLength,
        hash: shardHash,
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
      distribution: {
        concurrentDownloads: 1,
        maxRetries: 0,
      },
    },
  });

  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        estimate: async () => ({
          usage: 0,
          quota: 1024 * 1024 * 16,
        }),
      },
    },
    configurable: true,
  });

  globalThis.fetch = async (url) => {
    if (String(url).endsWith('/manifest.json')) {
      return new Response(JSON.stringify(createManifest()), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    if (String(url).endsWith('/model-00001-of-00001.bin')) {
      return new Response(shardBytes, {
        status: 200,
        headers: { 'content-length': String(shardBytes.byteLength) },
      });
    }
    if (String(url).endsWith('/tokenizer.json')) {
      return new Response('missing', {
        status: 404,
        statusText: 'Not Found',
      });
    }
    throw new Error(`Unexpected fetch url: ${url}`);
  };

  await assert.rejects(
    () => downloadModel('https://example.test/model', null, {
      requestPersist: false,
      concurrency: 1,
    }),
    /HTTP 404: Not Found/
  );
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

console.log('downloader-tokenizer-contract.test: ok');
