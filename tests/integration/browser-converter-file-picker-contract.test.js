import assert from 'node:assert/strict';

import { pickModelFiles } from '../../src/browser/browser-converter.js';

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
    createDirHandle('nested', [
      createFileHandle('model.safetensors'),
    ]),
    createFileHandle('tokenizer.model'),
  ]);

  const files = await pickModelFiles();
  assert.equal(files.length, 2);
  const nested = files.find((file) => file.name === 'model.safetensors');
  assert.ok(nested);
  assert.equal(nested.relativePath, 'nested/model.safetensors');
} finally {
  if (originalShowDirectoryPicker === undefined) {
    delete globalThis.showDirectoryPicker;
  } else {
    globalThis.showDirectoryPicker = originalShowDirectoryPicker;
  }
}

console.log('browser-converter-file-picker-contract.test: ok');
