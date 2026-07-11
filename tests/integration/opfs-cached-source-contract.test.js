import assert from 'node:assert/strict';

import { DEFAULT_KVCACHE_CONFIG, DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { resetRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import {
  cleanup,
  computeSHA256,
  deleteFileFromStore,
  openModelStore,
} from '../../src/storage/shard-manager.js';
import { ensureModelCachedSource } from '../../src/tooling/opfs-cache.js';

const originalFetch = globalThis.fetch;
const originalNavigator = globalThis.navigator;

function clone(value) {
  return structuredClone(value);
}

function toBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (typeof value === 'string') return new TextEncoder().encode(value);
  return new Uint8Array(0);
}

function createFileHandle(initialBytes = new Uint8Array(0)) {
  let bytes = initialBytes.slice(0);
  return {
    async getFile() {
      return {
        size: bytes.byteLength,
        async text() {
          return new TextDecoder().decode(bytes);
        },
        async arrayBuffer() {
          return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        },
        slice(start, end) {
          const sliced = bytes.slice(start, end);
          return {
            async arrayBuffer() {
              return sliced.buffer.slice(sliced.byteOffset, sliced.byteOffset + sliced.byteLength);
            },
          };
        },
      };
    },
    async createWritable() {
      let working = bytes.slice(0);
      return {
        async write(value) {
          if (value && typeof value === 'object' && value.type === 'write') {
            const payload = toBytes(value.data);
            const position = Number.isFinite(value.position) ? Math.max(0, Math.floor(value.position)) : 0;
            const next = new Uint8Array(Math.max(working.byteLength, position + payload.byteLength));
            next.set(working);
            next.set(payload, position);
            working = next;
            return;
          }
          working = toBytes(value).slice(0);
        },
        async close() {
          bytes = working;
        },
        async abort() {},
      };
    },
  };
}

function createDirectoryHandle() {
  const directories = new Map();
  const files = new Map();
  return {
    async getDirectoryHandle(name, options = {}) {
      if (!directories.has(name)) {
        if (options.create !== true) {
          const error = new Error(`Directory not found: ${name}`);
          error.name = 'NotFoundError';
          throw error;
        }
        directories.set(name, createDirectoryHandle());
      }
      return directories.get(name);
    },
    async getFileHandle(name, options = {}) {
      if (!files.has(name)) {
        if (options.create !== true) {
          const error = new Error(`File not found: ${name}`);
          error.name = 'NotFoundError';
          throw error;
        }
        files.set(name, createFileHandle());
      }
      return files.get(name);
    },
    async removeEntry(name) {
      if (files.delete(name) || directories.delete(name)) return;
      const error = new Error(`Entry not found: ${name}`);
      error.name = 'NotFoundError';
      throw error;
    },
    async *entries() {
      for (const [name, handle] of directories) yield [name, { kind: 'directory', ...handle }];
      for (const [name, handle] of files) yield [name, { kind: 'file', ...handle }];
    },
  };
}

async function createManifest(modelId, shardBytes) {
  return {
    version: 1,
    modelId,
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
          ...clone(DEFAULT_KVCACHE_CONFIG),
          layout: 'contiguous',
          kvDtype: 'f16',
          tiering: { ...clone(DEFAULT_KVCACHE_CONFIG).tiering, mode: 'off' },
          quantization: { ...clone(DEFAULT_KVCACHE_CONFIG).quantization },
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
    shards: [{
      index: 0,
      filename: 'model-00001-of-00001.bin',
      size: shardBytes.byteLength,
      hash: await computeSHA256(shardBytes),
      offset: 0,
    }],
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
  const root = createDirectoryHandle();
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async getDirectory() {
          return root;
        },
        async estimate() {
          return { usage: 0, quota: 1024 * 1024 };
        },
        async persisted() {
          return true;
        },
      },
    },
    configurable: true,
  });
  setRuntimeConfig({
    loading: {
      storage: {
        backend: {
          backend: 'opfs',
          opfs: {
            useSyncAccessHandle: false,
            maxConcurrentHandles: 2,
          },
        },
      },
    },
  });

  const modelId = 'opfs-cached-source-contract';
  const modelBaseUrl = 'https://example.test/opfs-cached-source-contract';
  const shardBytes = new Uint8Array([7, 11, 13, 17]);
  const manifest = await createManifest(modelId, shardBytes);
  const manifestText = JSON.stringify(manifest);
  const expectedManifestHash = await computeSHA256(new TextEncoder().encode(manifestText));
  let fetchCount = 0;
  globalThis.fetch = async (url) => {
    fetchCount += 1;
    const href = String(url);
    if (href === `${modelBaseUrl}/manifest.json`) {
      return new Response(manifestText, { status: 200 });
    }
    if (href === `${modelBaseUrl}/model-00001-of-00001.bin`) {
      return new Response(shardBytes, { status: 200 });
    }
    throw new Error(`Unexpected fetch: ${href}`);
  };

  const imported = await ensureModelCachedSource(modelId, modelBaseUrl, null, {
    expectedManifestHash,
  });
  assert.equal(imported.cacheState, 'imported');
  assert.equal(imported.storageBackend, 'opfs');
  assert.equal(imported.totalBytes, shardBytes.byteLength);
  assert.equal(imported.manifestText, manifestText);
  assert.equal(imported.manifestHash, expectedManifestHash);
  assert.deepEqual(new Uint8Array(await imported.storageContext.loadShard(0)), shardBytes);
  await imported.storageContext.close();
  assert.ok(fetchCount >= 2);

  globalThis.fetch = async () => {
    throw new Error('warm pinned cache must not use the network');
  };
  const hitEvents = [];
  const hit = await ensureModelCachedSource(modelId, modelBaseUrl, (event) => hitEvents.push(event), {
    expectedManifestHash,
  });
  assert.equal(hit.cacheState, 'verified-hit');
  assert.equal(hit.fromCache, true);
  assert.equal(hit.manifestText, manifestText);
  assert.equal(hit.manifestHash, expectedManifestHash);
  assert.deepEqual(new Uint8Array(await hit.storageContext.loadShardRange(0, 1, 2)), shardBytes.slice(1, 3));
  assert.ok(hitEvents.some((event) => event.stage === 'cache-hit'));
  await hit.storageContext.close();

  await openModelStore(modelId);
  await deleteFileFromStore('model-00001-of-00001.bin');
  let repairFetches = 0;
  globalThis.fetch = async (url) => {
    repairFetches += 1;
    const href = String(url);
    if (href === `${modelBaseUrl}/manifest.json`) return new Response(manifestText, { status: 200 });
    if (href === `${modelBaseUrl}/model-00001-of-00001.bin`) return new Response(shardBytes, { status: 200 });
    throw new Error(`Unexpected repair fetch: ${href}`);
  };
  const repaired = await ensureModelCachedSource(modelId, modelBaseUrl, null, {
    expectedManifestHash,
  });
  assert.equal(repaired.cacheState, 'imported');
  assert.ok(repairFetches >= 2);
  await repaired.storageContext.close();

  await assert.rejects(
    () => ensureModelCachedSource(modelId, modelBaseUrl, null, {
      expectedManifestHash: 'f'.repeat(64),
    }),
    /manifest hash mismatch/
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

console.log('opfs-cached-source-contract.test: ok');
