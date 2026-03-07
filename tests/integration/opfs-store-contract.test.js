import assert from 'node:assert/strict';

import { createOpfsStore } from '../../src/storage/backends/opfs-store.js';

const originalNavigator = globalThis.navigator;
const originalFileSystemSyncAccessHandle = globalThis.FileSystemSyncAccessHandle;

function createFileHandle() {
  return {
    async createSyncAccessHandle() {
      const error = new Error('not allowed');
      error.name = 'NotAllowedError';
      throw error;
    },
    async getFile() {
      return {
        async arrayBuffer() {
          return new Uint8Array([1, 2, 3]).buffer;
        },
      };
    },
  };
}

function createDirectoryHandle() {
  return {
    async getDirectoryHandle() {
      return createDirectoryHandle();
    },
    async getFileHandle() {
      return createFileHandle();
    },
  };
}

try {
  globalThis.FileSystemSyncAccessHandle = function MockSyncAccessHandle() {};
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async getDirectory() {
          return createDirectoryHandle();
        },
      },
    },
    configurable: true,
  });

  const store = createOpfsStore({
    opfsRootDir: 'models',
    useSyncAccessHandle: true,
    maxConcurrentHandles: 1,
  });
  await store.openModel('sync-contract', { create: true });
  await assert.rejects(
    () => store.readFile('manifest.json'),
    /explicitly requested but could not be opened/
  );
} finally {
  if (originalFileSystemSyncAccessHandle === undefined) {
    delete globalThis.FileSystemSyncAccessHandle;
  } else {
    globalThis.FileSystemSyncAccessHandle = originalFileSystemSyncAccessHandle;
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

console.log('opfs-store-contract.test: ok');
