import assert from 'node:assert/strict';
import { exportModelToDirectory } from '../../src/storage/export.js';
import { createIdbStore } from '../../src/storage/backends/idb-store.js';
import { selectStorageBackend } from '../../src/storage/backend-selection.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const originalConfig = structuredClone(getRuntimeConfig());
const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
const idbDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
let opens = 0;
let closes = 0;
let modelExists = true;
let listFails = false;
let opfsCalls = 0;
const destination = { getFileHandle() { throw new Error('Unexpected destination write'); } };
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
  storage: { async getDirectory() { opfsCalls++; throw new Error('Unexpected OPFS access'); } },
} });
globalThis.indexedDB = {
  open() {
    opens++;
    const request = { result: {
      close() { closes++; },
      transaction() {
        const tx = { objectStore() { return {
          get() {
            const read = { result: modelExists ? { value: true } : undefined };
            queueMicrotask(() => { read.onsuccess(); queueMicrotask(() => tx.oncomplete()); });
            return read;
          },
          openCursor() {
            const read = { error: new Error('list failed') };
            queueMicrotask(() => {
              if (listFails) read.onerror();
              else { read.onsuccess({ target: { result: null } }); queueMicrotask(() => tx.oncomplete()); }
            });
            return read;
          },
        }; } };
        return tx;
      },
    } };
    queueMicrotask(() => request.onsuccess());
    return request;
  },
};
try {
  setRuntimeConfig({ loading: { storage: { backend: { backend: 'memory' } } } });
  await assert.rejects(exportModelToDirectory('model', destination), /cannot reopen in-memory storage/);
  assert.equal(opfsCalls, 0);
  assert.equal(opens, 0);
  assert.throws(() => selectStorageBackend({ backend: 'typo' }, {}), /Unsupported storage backend/);

  setRuntimeConfig({ loading: { storage: { backend: { backend: 'indexeddb' } } } });
  const store = createIdbStore(getRuntimeConfig().loading.storage.backend.indexeddb);
  await store.init(); await store.cleanup(); await store.cleanup();
  assert.equal(closes, 1, 'cleanup closes once and is idempotent');
  await store.init(); await store.close();
  assert.equal(closes, 2, 'reinitialization owns a fresh connection');

  modelExists = false;
  await assert.rejects(store.openModelSession('missing', { create: false }), /Model not found/);
  assert.equal(opens, closes, 'failed session acquisition closes its connection');
  await assert.rejects(exportModelToDirectory('missing', destination), /Model not found/);
  assert.equal(opens, closes, 'failed export setup closes its connection');
  modelExists = true;
  await assert.rejects(exportModelToDirectory('model', destination, {
    onProgress() { throw new Error('progress failed'); },
  }), /progress failed/);
  assert.equal(opens, closes, 'progress callback failure closes connection');
  listFails = true;
  await assert.rejects(exportModelToDirectory('model', destination), /list failed/);
  assert.equal(opens, closes, 'listing failure closes connection');
  listFails = false;
  assert.deepEqual(await exportModelToDirectory('model', destination), { modelId: 'model', fileCount: 0 });
  assert.equal(opens, closes, 'successful export closes connection');
  await assert.rejects(exportModelToDirectory('model', destination, { chunkBytes: 0 }), /positive integer/);
} finally {
  setRuntimeConfig(originalConfig);
  if (navigatorDescriptor) Object.defineProperty(globalThis, 'navigator', navigatorDescriptor);
  else delete globalThis.navigator;
  if (idbDescriptor) Object.defineProperty(globalThis, 'indexedDB', idbDescriptor);
  else delete globalThis.indexedDB;
}
console.log('storage export and IndexedDB lifecycle tests passed');
