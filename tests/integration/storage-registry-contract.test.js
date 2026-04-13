import assert from 'node:assert/strict';

import { createIdbStore } from '../../src/storage/backends/idb-store.js';
import {
  loadModelRegistry,
  saveModelRegistry,
  registerModel,
} from '../../src/storage/registry.js';

function createRequest() {
  return {
    result: null,
    error: null,
    onsuccess: null,
    onerror: null,
  };
}

function createFakeIndexedDb() {
  const stores = {
    shards: new Map(),
    meta: new Map(),
  };

  function encodeKey(key) {
    return JSON.stringify(key);
  }

  function decodeKey(key) {
    return JSON.parse(key);
  }

  function matchesRange(key, range) {
    if (!range || !Array.isArray(range.lower) || !Array.isArray(range.upper)) {
      return false;
    }
    const tuple = Array.isArray(key) ? key : [key];
    for (let i = 0; i < tuple.length; i += 1) {
      if (tuple[i] < range.lower[i] || tuple[i] > range.upper[i]) {
        return false;
      }
    }
    return true;
  }

  function createStore(name, txState) {
    const map = stores[name];
    const schedule = (run) => {
      txState.pending += 1;
      queueMicrotask(() => {
        run();
        txState.pending -= 1;
        if (txState.pending === 0) {
          queueMicrotask(() => {
            txState.tx.oncomplete?.();
          });
        }
      });
    };
    return {
      put(value) {
        const request = createRequest();
        schedule(() => {
          const key = Object.prototype.hasOwnProperty.call(value, 'key')
            ? value.key
            : [value.modelId, value.filename, value.chunkIndex];
          map.set(encodeKey(key), structuredClone(value));
          request.result = key;
          request.onsuccess?.();
        });
        return request;
      },
      get(key) {
        const request = createRequest();
        schedule(() => {
          request.result = map.get(encodeKey(key)) ?? null;
          request.onsuccess?.();
        });
        return request;
      },
      delete(keyOrRange) {
        const request = createRequest();
        schedule(() => {
          if (keyOrRange && Array.isArray(keyOrRange.lower)) {
            for (const key of Array.from(map.keys())) {
              if (matchesRange(decodeKey(key), keyOrRange)) {
                map.delete(key);
              }
            }
          } else {
            map.delete(encodeKey(keyOrRange));
          }
          request.result = undefined;
          request.onsuccess?.();
        });
        return request;
      },
      openCursor() {
        const request = createRequest();
        const keys = Array.from(map.keys());
        let index = 0;
        const continueCursor = () => {
          if (index >= keys.length) {
            request.result = null;
            request.onsuccess?.({ target: request });
            return;
          }
          const key = keys[index];
          const value = map.get(key);
          index += 1;
          request.result = {
            key: decodeKey(key),
            value,
            continue: continueCursor,
          };
          request.onsuccess?.({ target: request });
        };
        schedule(() => {
          continueCursor();
        });
        return request;
      },
    };
  }

  return {
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
              contains(name) {
                return name === 'shards' || name === 'meta';
              },
            },
            createObjectStore() {
              return {
                createIndex() {
                  return;
                },
              };
            },
            transaction(_names, _mode) {
              const txState = {
                pending: 0,
                tx: {
                  error: null,
                  oncomplete: null,
                  onerror: null,
                  onabort: null,
                  objectStore(storeName) {
                    return createStore(storeName, txState);
                  },
                },
              };
              return txState.tx;
            },
          };
          request.result = db;
          request.onupgradeneeded?.({ target: request });
          request.onsuccess?.();
        });
        return request;
      },
    },
    IDBKeyRange: {
      bound(lower, upper) {
        return { lower, upper };
      },
    },
  };
}

const originalIndexedDb = globalThis.indexedDB;
const originalIdbKeyRange = globalThis.IDBKeyRange;
const originalNavigator = globalThis.navigator;

try {
  const fake = createFakeIndexedDb();
  globalThis.indexedDB = fake.indexedDB;
  globalThis.IDBKeyRange = fake.IDBKeyRange;
  Object.defineProperty(globalThis, 'navigator', {
    value: { storage: {} },
    configurable: true,
  });

  const savedEntry = await registerModel({
    modelId: 'registry-contract-model',
    backend: 'indexeddb',
  });
  assert.equal(savedEntry.backend, 'indexeddb');
  assert.equal(typeof savedEntry.savedAtUtc, 'string');

  const loaded = await loadModelRegistry();
  assert.equal(loaded.models.length, 1);
  assert.equal(loaded.models[0].modelId, 'registry-contract-model');
  assert.equal(typeof loaded.models[0].savedAtUtc, 'string');
  assert.match(loaded.models[0].savedAtUtc, /^\d{4}-\d{2}-\d{2}T/);

  const store = createIdbStore();
  await store.openModel('registry:models', { create: false });
  await store.writeFile('models.json', new TextEncoder().encode('{invalid'));
  await store.cleanup();

  await assert.rejects(
    () => loadModelRegistry(),
    /Expected property name|Unexpected token|JSON/
  );

  const legacyStore = createIdbStore();
  await legacyStore.openModel('registry:models', { create: true });
  await legacyStore.writeFile(
    'models.json',
    new TextEncoder().encode(JSON.stringify({
      models: [
        {
          modelId: 'legacy-created-at-only',
          backend: 'indexeddb',
          createdAt: '2026-04-13T00:00:00.000Z',
        },
      ],
    }))
  );
  await legacyStore.cleanup();

  const legacyLoaded = await loadModelRegistry();
  assert.equal(legacyLoaded.models[0].savedAtUtc, '2026-04-13T00:00:00.000Z');
} finally {
  if (originalIndexedDb === undefined) {
    delete globalThis.indexedDB;
  } else {
    globalThis.indexedDB = originalIndexedDb;
  }
  if (originalIdbKeyRange === undefined) {
    delete globalThis.IDBKeyRange;
  } else {
    globalThis.IDBKeyRange = originalIdbKeyRange;
  }
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
}

console.log('storage-registry-contract.test: ok');
