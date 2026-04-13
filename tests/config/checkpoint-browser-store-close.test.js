import assert from 'node:assert/strict';

import { loadCheckpoint, saveCheckpoint } from '../../src/experimental/training/checkpoint.js';

function createFakeIndexedDb() {
  const saved = new Map();
  let closeCount = 0;

  return {
    get closeCount() {
      return closeCount;
    },
    indexedDB: {
      open() {
        const request = {
          result: null,
          error: null,
          onsuccess: null,
          onerror: null,
          onupgradeneeded: null,
        };
        queueMicrotask(() => {
          const db = {
            objectStoreNames: {
              contains() {
                return false;
              },
            },
            createObjectStore() {},
            close() {
              closeCount += 1;
            },
            transaction(_storeName, mode) {
              const tx = {
                error: null,
                oncomplete: null,
                onerror: null,
                onabort: null,
                objectStore() {
                  return {
                    get(key) {
                      const getRequest = {
                        result: null,
                        error: null,
                        onsuccess: null,
                        onerror: null,
                      };
                      queueMicrotask(() => {
                        getRequest.result = saved.get(key) ?? null;
                        getRequest.onsuccess?.();
                      });
                      return getRequest;
                    },
                    put(value, key) {
                      saved.set(key, value);
                      queueMicrotask(() => {
                        tx.oncomplete?.();
                      });
                    },
                  };
                },
              };
              if (mode === 'readonly') {
                queueMicrotask(() => {
                  tx.oncomplete?.();
                });
              }
              return tx;
            },
          };
          request.result = db;
          request.onupgradeneeded?.();
          request.onsuccess?.();
        });
        return request;
      },
    },
  };
}

const originalIndexedDb = globalThis.indexedDB;

try {
  const fake = createFakeIndexedDb();
  globalThis.indexedDB = fake.indexedDB;

  await saveCheckpoint('browser-close-test', {
    weights: [1, 2, 3],
    metadata: {},
  });
  assert.equal(fake.closeCount, 1);

  const loaded = await loadCheckpoint('browser-close-test');
  assert.ok(loaded);
  assert.equal(fake.closeCount, 2);
} finally {
  if (originalIndexedDb === undefined) {
    delete globalThis.indexedDB;
  } else {
    globalThis.indexedDB = originalIndexedDb;
  }
}

console.log('checkpoint-browser-store-close.test: ok');
