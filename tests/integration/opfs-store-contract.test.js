import assert from 'node:assert/strict';

import { createOpfsStore } from '../../src/storage/backends/opfs-store.js';

const originalNavigator = globalThis.navigator;
const originalFileSystemSyncAccessHandle = globalThis.FileSystemSyncAccessHandle;

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  if (typeof data === 'string') {
    return new TextEncoder().encode(data);
  }
  return new Uint8Array(0);
}

function createFileHandle(initialBytes = new Uint8Array(0), options = {}) {
  let bytes = initialBytes.slice(0);
  return {
    async createSyncAccessHandle() {
      if (options.syncErrorName) {
        const error = new Error('not allowed');
        error.name = options.syncErrorName;
        throw error;
      }
      return {
        getSize() {
          return bytes.byteLength;
        },
        read(view, { at = 0 } = {}) {
          const slice = bytes.subarray(at, at + view.byteLength);
          view.set(slice);
          return slice.byteLength;
        },
        write(input, { at = 0 } = {}) {
          const payload = toUint8Array(input);
          const next = new Uint8Array(Math.max(bytes.byteLength, at + payload.byteLength));
          next.set(bytes, 0);
          next.set(payload, at);
          bytes = next;
          return payload.byteLength;
        },
        truncate(size) {
          bytes = bytes.slice(0, size);
        },
        flush() {},
        close() {},
      };
    },
    async getFile() {
      return {
        size: bytes.byteLength,
        async text() {
          return new TextDecoder().decode(bytes);
        },
        async arrayBuffer() {
          return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        },
      };
    },
    async createWritable() {
      let working = bytes.slice(0);
      return {
        async write(value) {
          if (value && typeof value === 'object' && value.type === 'write') {
            const payload = toUint8Array(value.data);
            const position = Number.isFinite(value.position) ? Math.max(0, Math.floor(value.position)) : 0;
            const next = new Uint8Array(Math.max(working.byteLength, position + payload.byteLength));
            next.set(working, 0);
            next.set(payload, position);
            working = next;
            return;
          }
          working = toUint8Array(value).slice(0);
        },
        async close() {
          bytes = working;
        },
        async abort() {},
      };
    },
  };
}

function createDirectoryHandle(options = {}) {
  const directories = new Map();
  const files = new Map();
  return {
    async getDirectoryHandle(name, handleOptions = {}) {
      if (!directories.has(name)) {
        if (handleOptions.create !== true) {
          const error = new Error(`Directory not found: ${name}`);
          error.name = 'NotFoundError';
          throw error;
        }
        directories.set(name, createDirectoryHandle(options));
      }
      return directories.get(name);
    },
    async getFileHandle(name, handleOptions = {}) {
      if (!files.has(name)) {
        if (handleOptions.create !== true) {
          const error = new Error(`File not found: ${name}`);
          error.name = 'NotFoundError';
          throw error;
        }
        files.set(name, createFileHandle(new Uint8Array(0), options));
      }
      return files.get(name);
    },
    async removeEntry(name) {
      if (files.delete(name)) {
        return;
      }
      if (directories.delete(name)) {
        return;
      }
      const error = new Error(`Entry not found: ${name}`);
      error.name = 'NotFoundError';
      throw error;
    },
    async *entries() {
      for (const [name, handle] of directories.entries()) {
        yield [name, { kind: 'directory', ...handle }];
      }
      for (const [name, handle] of files.entries()) {
        yield [name, { kind: 'file', ...handle }];
      }
    },
  };
}

try {
  globalThis.FileSystemSyncAccessHandle = function MockSyncAccessHandle() {};
  const syncRoot = createDirectoryHandle({ syncErrorName: 'NotAllowedError' });
  const syncModelsDir = await syncRoot.getDirectoryHandle('models', { create: true });
  const syncModelDir = await syncModelsDir.getDirectoryHandle('sync-contract', { create: true });
  await syncModelDir.getFileHandle('manifest.json', { create: true });
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async getDirectory() {
          return syncRoot;
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

  const root = createDirectoryHandle();
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async getDirectory() {
          return root;
        },
      },
    },
    configurable: true,
  });

  const nestedStore = createOpfsStore({
    opfsRootDir: 'models',
    useSyncAccessHandle: false,
    maxConcurrentHandles: 1,
  });
  await nestedStore.openModel('nested-contract', { create: true });
  await nestedStore.writeFile('weights/model.safetensors', new Uint8Array([1, 2, 3, 4]));
  await nestedStore.writeFile('tokenizers/main/tokenizer.json', new TextEncoder().encode('{"ok":true}'));

  const nestedBytes = new Uint8Array(await nestedStore.readFile('weights/model.safetensors'));
  assert.deepEqual(Array.from(nestedBytes), [1, 2, 3, 4]);

  const listed = await nestedStore.listFiles();
  assert.deepEqual(
    listed.sort((left, right) => left.localeCompare(right)),
    ['tokenizers/main/tokenizer.json', 'weights/model.safetensors']
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
