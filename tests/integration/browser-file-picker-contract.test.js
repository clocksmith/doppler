import assert from 'node:assert/strict';

import { pickModelDirectory } from '../../src/browser/file-picker.js';

function createFileHandle(name) {
  return {
    kind: 'file',
    name,
    async getFile() {
      return { name };
    },
  };
}

function createDirHandle(name, entries) {
  return {
    kind: 'directory',
    name,
    async *values() {
      for (const entry of entries) {
        yield entry;
      }
    },
  };
}

const originalShowDirectoryPicker = globalThis.showDirectoryPicker;

try {
  globalThis.showDirectoryPicker = async () => createDirHandle('root', [
    createDirHandle('a', [
      createDirHandle('b', [
        createDirHandle('c', [
          createDirHandle('d', [
            createDirHandle('e', [
              createFileHandle('model.safetensors'),
            ]),
          ]),
        ]),
      ]),
    ]),
  ]);

  await assert.rejects(
    () => pickModelDirectory(),
    /Model directory exceeds supported depth \(4\) near "a\/b\/c\/d\/e"/
  );
} finally {
  if (originalShowDirectoryPicker === undefined) {
    delete globalThis.showDirectoryPicker;
  } else {
    globalThis.showDirectoryPicker = originalShowDirectoryPicker;
  }
}

console.log('browser-file-picker-contract.test: ok');
