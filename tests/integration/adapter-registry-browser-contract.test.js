import assert from 'node:assert/strict';

import {
  AdapterRegistry,
  createMemoryRegistry,
} from '../../src/experimental/adapters/adapter-registry.js';

const originalIndexedDb = globalThis.indexedDB;
const originalWindow = globalThis.window;

try {
  delete globalThis.indexedDB;
  globalThis.window = {};

  assert.throws(
    () => new AdapterRegistry(),
    /requires IndexedDB in browser environments/
  );

  const memoryRegistry = createMemoryRegistry();
  assert.ok(memoryRegistry instanceof AdapterRegistry);
} finally {
  if (originalIndexedDb === undefined) {
    delete globalThis.indexedDB;
  } else {
    globalThis.indexedDB = originalIndexedDb;
  }
  if (originalWindow === undefined) {
    delete globalThis.window;
  } else {
    globalThis.window = originalWindow;
  }
}

console.log('adapter-registry-browser-contract.test: ok');
