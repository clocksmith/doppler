import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { downloadModel } from '../../src/storage/downloader.js';
import { cleanup } from '../../src/storage/shard-manager.js';
import { ensureModelCachedSource } from '../../src/tooling/opfs-cache.js';

const originalNavigator = globalThis.navigator;
const originalFetch = globalThis.fetch;

let fetchCalls = 0;

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
        maxRetries: 0,
      },
    },
  });

  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        estimate: async () => ({
          usage: 0,
          quota: 1024 * 1024,
        }),
      },
    },
    configurable: true,
  });

  globalThis.fetch = async () => {
    fetchCalls += 1;
    throw new Error('fetch should not be called after abort');
  };

  const controller = new AbortController();
  controller.abort();

  await assert.rejects(
    () => downloadModel('https://example.test/model', null, {
      requestPersist: false,
      signal: controller.signal,
    }),
    /Download aborted/
  );

  assert.equal(fetchCalls, 0);

  const manifestController = new AbortController();
  let manifestFetchSignal = null;
  globalThis.fetch = async (_url, init = {}) => {
    fetchCalls += 1;
    manifestFetchSignal = init.signal || null;
    manifestController.abort();
    if (init.signal?.aborted) {
      const error = new Error('manifest fetch aborted');
      error.name = 'AbortError';
      throw error;
    }
    throw new Error('manifest transport continued after cancellation');
  };

  await assert.rejects(
    () => downloadModel('https://example.test/model', null, {
      requestPersist: false,
      signal: manifestController.signal,
    }),
    (error) => error?.name === 'AbortError'
  );
  assert.equal(manifestFetchSignal, manifestController.signal);

  fetchCalls = 0;
  const cacheController = new AbortController();
  cacheController.abort();
  await assert.rejects(
    () => ensureModelCachedSource(
      'cancelled-cache-model',
      'https://example.test/cancelled-cache-model',
      null,
      {
        expectedManifestHash: 'f'.repeat(64),
        signal: cacheController.signal,
      }
    ),
    (error) => error?.name === 'AbortError'
  );
  assert.equal(fetchCalls, 0);
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

console.log('downloader-external-signal-contract.test: ok');
