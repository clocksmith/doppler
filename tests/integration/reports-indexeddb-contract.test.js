import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { createIdbStore } from '../../src/storage/backends/idb-store.js';
import { saveReport } from '../../src/storage/reports.js';

function createFakeIndexedDb() {
  const databases = new Map();

  function getStore(database, name) {
    if (!database.stores.has(name)) {
      database.stores.set(name, new Map());
    }
    return database.stores.get(name);
  }

  function serializeKey(key) {
    return JSON.stringify(key);
  }

  function createRequest(result = null) {
    return {
      result,
      error: null,
      onsuccess: null,
      onerror: null,
    };
  }

  function queueSuccess(request, result) {
    queueMicrotask(() => {
      request.result = result;
      request.onsuccess?.({ target: request });
    });
  }

  return {
    indexedDB: {
      open(dbName) {
        const request = {
          result: null,
          error: null,
          onsuccess: null,
          onerror: null,
          onupgradeneeded: null,
        };

        queueMicrotask(() => {
          const database = databases.get(dbName) ?? {
            stores: new Map(),
            objectStoreNames: {
              contains(name) {
                return databases.get(dbName)?.stores.has(name) ?? false;
              },
            },
            createObjectStore(name) {
              const store = getStore(database, name);
              return {
                createIndex() {
                  return store;
                },
              };
            },
            transaction(storeNames) {
              const names = Array.isArray(storeNames) ? storeNames : [storeNames];
              let completed = false;
              const finish = () => {
                if (completed) {
                  return;
                }
                completed = true;
                setTimeout(() => {
                  tx.oncomplete?.();
                }, 0);
              };
              const tx = {
                error: null,
                oncomplete: null,
                onerror: null,
                onabort: null,
                objectStore(name) {
                  if (!names.includes(name)) {
                    throw new Error(`Unknown store: ${name}`);
                  }
                  const backing = getStore(database, name);
                  return {
                    get(key) {
                      const req = createRequest();
                      queueMicrotask(() => {
                        queueSuccess(req, backing.get(serializeKey(key)) ?? null);
                        finish();
                      });
                      return req;
                    },
                    put(value) {
                      backing.set(
                        serializeKey(name === 'meta' && value?.key !== undefined ? value.key : (
                          name === 'shards'
                            ? [value.modelId, value.filename, value.chunkIndex]
                            : value?.id
                        )),
                        value
                      );
                      finish();
                    },
                    delete(key) {
                      if (key && typeof key === 'object' && 'lower' in key && 'upper' in key) {
                        for (const entryKey of Array.from(backing.keys())) {
                          const parsed = JSON.parse(entryKey);
                          if (Array.isArray(parsed)) {
                            const [modelId, filename] = parsed;
                            if (
                              modelId === key.lower[0]
                              && filename >= key.lower[1]
                              && filename <= key.upper[1]
                            ) {
                              backing.delete(entryKey);
                            }
                          }
                        }
                      } else {
                        backing.delete(serializeKey(key));
                      }
                      finish();
                    },
                  };
                },
              };
              return tx;
            },
            close() {},
          };

          databases.set(dbName, database);
          request.result = database;
          request.onupgradeneeded?.({ target: request });
          request.onsuccess?.({ target: request });
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
const originalIDBKeyRange = globalThis.IDBKeyRange;
const originalNavigator = globalThis.navigator;

try {
  const fake = createFakeIndexedDb();
  globalThis.indexedDB = fake.indexedDB;
  globalThis.IDBKeyRange = fake.IDBKeyRange;
  Object.defineProperty(globalThis, 'navigator', {
    value: {},
    configurable: true,
  });

  setRuntimeConfig({
    loading: {
      storage: {
        backend: {
          backend: 'indexeddb',
        },
      },
    },
  });

  const timestamp = '2026-03-02T00:00:00.000Z';
  const saved = await saveReport(
    'idb-report',
    { suite: 'storage', passed: 1 },
    { timestamp }
  );
  assert.equal(saved.backend, 'indexeddb');

  const filename = saved.path.split('/').pop();
  const store = createIdbStore({});
  await store.openModel('reports:idb-report', { create: false });
  const raw = await store.readFile(filename);
  await store.cleanup();

  const parsed = JSON.parse(new TextDecoder().decode(raw));
  assert.equal(parsed.suite, 'storage');
  assert.equal(parsed.passed, 1);
} finally {
  resetRuntimeConfig();
  if (originalIndexedDb === undefined) {
    delete globalThis.indexedDB;
  } else {
    globalThis.indexedDB = originalIndexedDb;
  }
  if (originalIDBKeyRange === undefined) {
    delete globalThis.IDBKeyRange;
  } else {
    globalThis.IDBKeyRange = originalIDBKeyRange;
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

console.log('reports-indexeddb-contract.test: ok');
