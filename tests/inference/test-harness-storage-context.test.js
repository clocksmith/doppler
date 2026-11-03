import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { getRuntimeConfig, setRuntimeConfig, snapshotRuntimeConfig } from '../../src/config/runtime.js';

import {
  createHarnessShardStorageContext,
} from '../../src/tooling/test-harness.js';

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

{
  const storageContext = createHarnessShardStorageContext(
    'https://example.com/model',
    manifest,
    null,
    { loadMode: 'http' }
  );
  assert.equal(typeof storageContext.loadShard, 'function');
  assert.equal(typeof storageContext.loadShardRange, 'function');
  assert.equal(typeof storageContext.streamShardRange, 'function');
}

// The shared HTTP harness consumes the resolved verification budget instead of
// silently leaving the runtime field unused. A multi-chunk source distinguishes
// buffered verification from the incremental path.
{
  const originalConfig = snapshotRuntimeConfig(getRuntimeConfig());
  const originalFetch = globalThis.fetch;
  const bytes = Uint8Array.from({ length: 4 * 1024 * 1024 + 73 }, (_, index) => index % 251);
  const ranges = [];
  try {
    setRuntimeConfig({ loading: { shardCache: { verifyHashes: true, verificationBufferBytes: bytes.length } } });
    globalThis.fetch = async (_url, options) => {
      const [, startText, endText] = /^bytes=(\d+)-(\d+)$/.exec(options.headers.Range);
      const start = Number(startText), end = Number(endText);
      ranges.push([start, end]);
      return { ok: true, status: 206, arrayBuffer: async () => bytes.slice(start, end + 1).buffer };
    };
    const context = createHarnessShardStorageContext('https://example.com/model', {
      modelId: 'harness-buffered-verification', hashAlgorithm: 'sha256', shards: [
        { filename: 'weights.bin', size: bytes.length, hash: createHash('sha256').update(bytes).digest('hex') },
      ],
    }, null, { loadMode: 'http' });
    assert.deepEqual(new Uint8Array(await context.loadShardRange(0, 5, 3)), bytes.slice(5, 8));
    assert.deepEqual(ranges, [[0, bytes.length - 1], [5, 7]]);
  } finally {
    globalThis.fetch = originalFetch;
    setRuntimeConfig(originalConfig);
  }
}

console.log('test-harness-storage-context.test: ok');
