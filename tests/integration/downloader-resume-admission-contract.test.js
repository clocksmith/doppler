import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { parseManifest } from '../../src/formats/rdrr/index.js';
import { resetRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import {
  cleanup,
  computeHash,
  openModelStore,
  saveAuxFile,
  writeShard,
} from '../../src/storage/shard-manager.js';
import {
  downloadModel,
  inspectModelDownloadResume,
} from '../../src/storage/downloader.js';
import { buildManifestVersionSet } from '../../src/storage/download/integrity.js';
import { saveDownloadState } from '../../src/storage/download/state.js';
import { createExecutionContractSession } from '../helpers/execution-v1-fixtures.js';

function createFakeIndexedDb() {
  const databases = new Map();
  const createRequest = () => ({ result: null, error: null, onsuccess: null, onerror: null });
  const complete = (request, result) => queueMicrotask(() => {
    request.result = result;
    request.onsuccess?.({ target: request });
  });

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
              if (!records) throw new Error(`Unknown store: ${storeName}`);
              return {
                error: null,
                onerror: null,
                onabort: null,
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

function clone(value) {
  return structuredClone(value);
}

async function createManifest(modelId, shardPayloads) {
  const shards = [];
  let offset = 0;
  for (const [index, payload] of shardPayloads.entries()) {
    shards.push({
      index,
      filename: `shard-${index}.bin`,
      size: payload.byteLength,
      hash: await computeHash(payload, 'sha256'),
      offset,
    });
    offset += payload.byteLength;
  }
  return {
    version: 1,
    modelId,
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256',
    totalSize: offset,
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
    shards,
    groups: {
      layers: {
        type: 'layer',
        version: '1.0.0',
        shards: shards.map((shard) => shard.index),
        tensors: [],
        hash: '0'.repeat(64),
      },
    },
  };
}

async function seedCompletedShard(modelId, manifest, payload, replacement = payload) {
  parseManifest(JSON.stringify(manifest));
  await openModelStore(modelId);
  await writeShard(0, payload);
  if (replacement !== payload) {
    await saveAuxFile(manifest.shards[0].filename, replacement);
  }
  await saveDownloadState({
    modelId,
    baseUrl: `https://example.test/${modelId}`,
    manifest,
    manifestVersionSet: buildManifestVersionSet(manifest),
    completedShards: new Set([0]),
    startTime: 1,
    status: 'paused',
  });
}

const originalFetch = globalThis.fetch;
const originalIndexedDb = globalThis.indexedDB;
const originalNavigator = globalThis.navigator;

try {
  const shardA = new Uint8Array([1, 2, 3, 4]);
  const shardB = new Uint8Array([5, 6, 7, 8]);
  const manifest = await createManifest('resume-admission-model', [shardA, shardB]);
  let availableBytes = 1024;

  globalThis.indexedDB = createFakeIndexedDb();
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async estimate() {
          return { usage: 100, quota: 100 + availableBytes };
        },
      },
    },
    configurable: true,
  });
  setRuntimeConfig({
    loading: {
      storage: {
        backend: {
          backend: 'memory',
          memory: { maxBytes: 1024 * 1024 },
        },
      },
      distribution: {
        concurrentDownloads: 1,
        maxRetries: 0,
      },
    },
  });

  await seedCompletedShard(manifest.modelId, manifest, shardA);
  const inspection = await inspectModelDownloadResume(manifest.modelId, manifest);
  assert.equal(inspection.schemaVersion, 'doppler.model-download-resume-inspection.v1');
  assert.equal(inspection.manifestMatched, true);
  assert.equal(inspection.totalBytes, 8);
  assert.equal(inspection.verifiedBytes, 4);
  assert.equal(inspection.remainingBytes, 4);
  assert.equal(inspection.verifiedShards, 1);

  availableBytes = 6;
  const fetchedShardPaths = [];
  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href.endsWith('/manifest.json')) {
      return new Response(JSON.stringify(manifest), { status: 200 });
    }
    if (href.endsWith('/shard-1.bin')) {
      fetchedShardPaths.push('shard-1.bin');
      return new Response(shardB, {
        status: 200,
        headers: { 'content-length': String(shardB.byteLength) },
      });
    }
    throw new Error(`Unexpected fetch: ${href}`);
  };
  assert.equal(await downloadModel(`https://example.test/${manifest.modelId}`, null, {
    requestPersist: false,
    concurrency: 1,
  }), true);
  assert.deepEqual(fetchedShardPaths, ['shard-1.bin']);

  availableBytes = 1024;
  const corruptManifest = await createManifest('corrupt-resume-admission-model', [shardA, shardB]);
  await seedCompletedShard(
    corruptManifest.modelId,
    corruptManifest,
    shardA,
    new Uint8Array([9, 9, 9, 9])
  );
  const corruptInspection = await inspectModelDownloadResume(corruptManifest.modelId, corruptManifest);
  assert.equal(corruptInspection.manifestMatched, true);
  assert.equal(corruptInspection.verifiedBytes, 0);
  assert.equal(corruptInspection.remainingBytes, 8);
  assert.equal(corruptInspection.verifiedShards, 0);
} finally {
  globalThis.fetch = originalFetch;
  if (originalIndexedDb === undefined) delete globalThis.indexedDB;
  else globalThis.indexedDB = originalIndexedDb;
  if (originalNavigator === undefined) delete globalThis.navigator;
  else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
  resetRuntimeConfig();
  await cleanup();
}

console.log('downloader-resume-admission-contract.test: ok');
