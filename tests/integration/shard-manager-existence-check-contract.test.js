import assert from 'node:assert/strict';

import {
  checkFileExistsInBackend,
  getFileSizeInBackend,
} from '../../src/storage/shard-manager.js';

{
  let getFileSizeCalls = 0;
  let readFileCalls = 0;
  const backend = {
    async getFileSize(filename) {
      getFileSizeCalls += 1;
      assert.equal(filename, 'model-00001-of-00001.bin');
      return 1234;
    },
    async readFile() {
      readFileCalls += 1;
      throw new Error('readFile should not be used when getFileSize is available');
    },
  };

  const exists = await checkFileExistsInBackend(backend, 'model-00001-of-00001.bin');
  assert.equal(exists, true);
  const size = await getFileSizeInBackend(backend, 'model-00001-of-00001.bin');
  assert.equal(size, 1234);
  assert.equal(getFileSizeCalls, 2);
  assert.equal(readFileCalls, 0);
}

{
  let getFileSizeCalls = 0;
  let readFileCalls = 0;
  const backend = {
    async getFileSize() {
      getFileSizeCalls += 1;
      const error = new Error('missing');
      error.name = 'NotFoundError';
      throw error;
    },
    async readFile() {
      readFileCalls += 1;
      return new ArrayBuffer(0);
    },
  };

  const exists = await checkFileExistsInBackend(backend, 'missing.bin');
  assert.equal(exists, false);
  const size = await getFileSizeInBackend(backend, 'missing.bin');
  assert.equal(size, null);
  assert.equal(getFileSizeCalls, 2);
  assert.equal(readFileCalls, 0);
}

{
  let readFileCalls = 0;
  const backend = {
    async readFile(filename) {
      readFileCalls += 1;
      assert.equal(filename, 'fallback.bin');
      return new ArrayBuffer(16);
    },
  };

  const exists = await checkFileExistsInBackend(backend, 'fallback.bin');
  assert.equal(exists, true);
  const size = await getFileSizeInBackend(backend, 'fallback.bin');
  assert.equal(size, 16);
  assert.equal(readFileCalls, 2);
}

console.log('shard-manager-existence-check-contract.test: ok');
