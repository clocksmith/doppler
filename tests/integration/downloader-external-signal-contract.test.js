import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { downloadModel } from '../../src/storage/downloader.js';
import { cleanup } from '../../src/storage/shard-manager.js';

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
