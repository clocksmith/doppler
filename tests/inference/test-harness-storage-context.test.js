import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import {
  createHarnessShardStorageContext,
} from '../../src/inference/test-harness.js';

const manifest = {
  modelId: 'test-harness-storage-context',
  version: 1,
  hashAlgorithm: 'blake3',
  shards: [
    {
      filename: 'shard_00000.bin',
      size: 4,
      hash: '',
    },
  ],
};

{
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-harness-storage-'));
  try {
    await fs.writeFile(path.join(tempRoot, 'shard_00000.bin'), new Uint8Array([1, 2, 3, 4]));
    const storageContext = createHarnessShardStorageContext(pathToFileURL(tempRoot).href, manifest);
    assert.equal(typeof storageContext.loadShard, 'function');
    assert.equal(typeof storageContext.loadShardRange, 'function');
    assert.equal(typeof storageContext.streamShardRange, 'function');
    const slice = new Uint8Array(await storageContext.loadShardRange(0, 1, 2));
    assert.deepEqual(Array.from(slice), [2, 3]);
  } finally {
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
}

{
  const storageContext = createHarnessShardStorageContext('https://example.com/model', manifest);
  assert.equal(typeof storageContext.loadShard, 'function');
  assert.equal(storageContext.loadShardRange, undefined);
  assert.equal(storageContext.streamShardRange, undefined);
}

console.log('test-harness-storage-context.test: ok');
