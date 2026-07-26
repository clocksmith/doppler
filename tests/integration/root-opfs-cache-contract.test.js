import assert from 'node:assert/strict';

import { resolvePersistentBrowserLoadSource } from '../../src/client/runtime/index.js';

const manifest = {
  modelId: 'root-opfs-cache-contract',
  hashAlgorithm: 'sha256',
  shards: [],
};
const manifestText = JSON.stringify(manifest);
const loadSource = {
  modelId: manifest.modelId,
  baseUrl: 'https://example.test/root-opfs-cache-contract',
  manifest,
  manifestText,
  storageBaseUrl: 'https://example.test/root-opfs-cache-contract',
  storageManifest: manifest,
  storageManifestText: manifestText,
  trace: [],
};

assert.equal(
  await resolvePersistentBrowserLoadSource(loadSource, false),
  loadSource
);

await assert.rejects(
  () => resolvePersistentBrowserLoadSource(loadSource, 'opfs'),
  /browser-only/
);

await assert.rejects(
  () => resolvePersistentBrowserLoadSource(loadSource, 'memory'),
  /must be false or "opfs"/
);

const processDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'process');
const events = [];
const storageContext = {
  async close() {},
};
let request = null;

try {
  Object.defineProperty(globalThis, 'process', {
    value: undefined,
    configurable: true,
    writable: true,
  });
  const resolved = await resolvePersistentBrowserLoadSource(
    loadSource,
    'opfs',
    (event) => events.push(event),
    async (modelId, modelBaseUrl, onProgress, options) => {
      request = { modelId, modelBaseUrl, options };
      onProgress({
        stage: 'cache-hit',
        percent: 100,
        message: 'Verified OPFS cache hit',
      });
      return {
        storageContext,
        storageBackend: 'opfs',
        cacheState: 'verified-hit',
        fromCache: true,
        manifestHash: options.expectedManifestHash,
        totalBytes: 4096,
      };
    }
  );

  assert.equal(request.modelId, manifest.modelId);
  assert.equal(request.modelBaseUrl, loadSource.storageBaseUrl);
  assert.match(request.options.expectedManifestHash, /^[a-f0-9]{64}$/);
  assert.equal(resolved.storageContext, storageContext);
  assert.equal(resolved.storage, storageContext);
  assert.deepEqual(resolved.persistentCache, {
    backend: 'opfs',
    state: 'verified-hit',
    fromCache: true,
    manifestHash: request.options.expectedManifestHash,
    totalBytes: 4096,
  });
  assert.deepEqual(events, [{
    phase: 'cache',
    percent: 25,
    message: 'Verified OPFS cache hit',
  }]);
} finally {
  if (processDescriptor) {
    Object.defineProperty(globalThis, 'process', processDescriptor);
  } else {
    delete globalThis.process;
  }
}

console.log('root-opfs-cache-contract.test: ok');
