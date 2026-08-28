import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { computeHash, cleanup } from '../../src/storage/shard-manager.js';
import { downloadModel, getDownloadProgress } from '../../src/storage/downloader.js';
import { createExecutionContractSession } from '../helpers/execution-v1-fixtures.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function createFakeIndexedDb() {
  const databases = new Map();
  const complete = (request, result) => queueMicrotask(() => {
    request.result = result;
    request.onsuccess?.({ target: request });
  });
  const createRequest = () => ({ result: null, error: null, onsuccess: null, onerror: null });
  return {
    open(name) {
      const request = {
        result: null,
        error: null,
        onsuccess: null,
        onerror: null,
        onupgradeneeded: null,
      };
      queueMicrotask(() => {
        let database = databases.get(name);
        const created = !database;
        if (!database) {
          const stores = new Map();
          database = {
            objectStoreNames: { contains: (storeName) => stores.has(storeName) },
            createObjectStore(storeName) {
              const records = new Map();
              stores.set(storeName, records);
              return { createIndex() {} };
            },
            transaction(storeName) {
              const records = stores.get(storeName);
              return {
                objectStore() {
                  return {
                    put(value) {
                      const operation = createRequest();
                      records.set(String(value.modelId), structuredClone(value));
                      complete(operation, value.modelId);
                      return operation;
                    },
                    get(key) {
                      const operation = createRequest();
                      complete(operation, structuredClone(records.get(String(key)) ?? null));
                      return operation;
                    },
                    delete(key) {
                      const operation = createRequest();
                      records.delete(String(key));
                      complete(operation, undefined);
                      return operation;
                    },
                    getAll() {
                      const operation = createRequest();
                      complete(operation, Array.from(records.values(), (value) => structuredClone(value)));
                      return operation;
                    },
                  };
                },
              };
            },
          };
          databases.set(name, database);
        }
        request.result = database;
        if (created) request.onupgradeneeded?.({ target: request });
        request.onsuccess?.({ target: request });
      });
      return request;
    },
  };
}

const originalNavigator = globalThis.navigator;
const originalFetch = globalThis.fetch;
const originalIndexedDb = globalThis.indexedDB;

const shardBytes = new Uint8Array([1, 2, 3, 4]);
const shardHash = await computeHash(shardBytes, 'sha256');
const shardBlake3 = await computeHash(shardBytes, 'blake3');

function createManifest(hashAlgorithm = 'sha256') {
  return {
    version: 1,
    modelId: 'manifest-model-id',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm,
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
      session: createExecutionContractSession(),
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
        blake3: shardBlake3,
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
  globalThis.indexedDB = createFakeIndexedDb();
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
      return new Response(JSON.stringify(createManifest(globalThis.__testManifestHashAlgorithm || 'sha256')), {
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

  for (const hashAlgorithm of ['sha256', 'blake3']) {
    globalThis.__testManifestHashAlgorithm = hashAlgorithm;
    await assert.rejects(
      () => downloadModel('https://example.test/model', null, {
        requestPersist: false,
        concurrency: 1,
      }),
      /HTTP 404: Not Found/
    );
    await cleanup();
  }

  const controller = new AbortController();
  let tokenizerFetchSignal = null;
  globalThis.fetch = async (url, init = {}) => {
    if (String(url).endsWith('/manifest.json')) {
      return new Response(JSON.stringify(createManifest('sha256')), {
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
      tokenizerFetchSignal = init.signal || null;
      controller.abort();
      if (init.signal?.aborted) {
        const error = new Error('tokenizer fetch aborted');
        error.name = 'AbortError';
        throw error;
      }
      return new Response('{}', { status: 200 });
    }
    throw new Error(`Unexpected fetch url: ${url}`);
  };

  await assert.rejects(
    () => downloadModel('https://example.test/model', null, {
      requestPersist: false,
      concurrency: 1,
      signal: controller.signal,
    }),
    (error) => error?.name === 'AbortError'
  );
  assert.equal(tokenizerFetchSignal?.aborted, true);
  assert.equal((await getDownloadProgress('manifest-model-id'))?.status, 'paused');
} finally {
  globalThis.fetch = originalFetch;
  if (originalIndexedDb === undefined) delete globalThis.indexedDB;
  else globalThis.indexedDB = originalIndexedDb;
  delete globalThis.__testManifestHashAlgorithm;
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
  resetRuntimeConfig();
}

console.log('downloader-tokenizer-contract.test: ok');
