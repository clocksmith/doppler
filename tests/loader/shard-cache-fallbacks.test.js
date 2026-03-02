import assert from 'node:assert/strict';

const { ShardCache } = await import('../../src/loader/shard-cache.js');

function createLoadingConfig() {
  return {
    verifyHashes: false,
    maxConcurrentLoads: 0,
    opfsEntries: 4,
    networkEntries: 4,
    moeMaxEntries: 8,
  };
}

function createManifest(size = 8) {
  return {
    modelId: 'shard-cache-fallbacks-test',
    hashAlgorithm: 'sha256',
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size,
        hash: '0'.repeat(64),
        hashAlgorithm: 'sha256',
      },
    ],
  };
}

async function collectStream(iterable) {
  const out = [];
  for await (const chunk of iterable) {
    out.push(...chunk);
  }
  return out;
}

{
  const shard = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const cache = new ShardCache({
    maxEntries: 2,
    loadingConfig: createLoadingConfig(),
    verifyHashes: false,
    manifest: createManifest(shard.byteLength),
    customLoader: async () => shard,
    customRangeLoader: async () => {
      const error = new Error('range not supported');
      error.code = 'not_supported';
      throw error;
    },
  });

  const range = await cache.loadRange(0, 2, 3);
  assert.deepEqual(Array.from(new Uint8Array(range)), [2, 3, 4]);
  assert.equal(cache.lastSource?.mode, 'range');
  assert.equal(cache.lastSource?.path, 'custom-loader-slice');
  assert.equal(cache.lastSource?.fallback, 'custom_range_not_supported');
}

{
  const shard = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const cache = new ShardCache({
    maxEntries: 2,
    loadingConfig: createLoadingConfig(),
    verifyHashes: false,
    manifest: createManifest(shard.byteLength),
    customStreamLoader: async function* () {
      yield shard.slice(0, 2);
      const error = new Error('stream interrupted');
      error.code = 'stream_interrupted';
      throw error;
    },
    customRangeLoader: async (_index, offset, length) => {
      const start = Math.max(0, Number.isFinite(offset) ? Math.floor(offset) : 0);
      const size = Math.max(0, Number.isFinite(length) ? Math.floor(length) : 0);
      return shard.slice(start, start + size);
    },
  });

  const bytes = await collectStream(cache.streamRange(0, 0, 6, { chunkBytes: 2 }));
  assert.deepEqual(bytes, [0, 1, 2, 3, 4, 5]);
  assert.equal(cache.lastSource?.mode, 'stream');
  assert.equal(cache.lastSource?.path, 'custom-range-fallback');
  assert.equal(cache.lastSource?.fallback, 'custom_stream_interrupted_resume');
}

{
  const shard = new Uint8Array([0, 1, 2, 3]);
  let rangeCall = 0;
  const cache = new ShardCache({
    maxEntries: 2,
    loadingConfig: createLoadingConfig(),
    verifyHashes: false,
    manifest: createManifest(shard.byteLength),
    customRangeLoader: async (_index, offset, length) => {
      rangeCall++;
      if (rangeCall === 2) {
        return new Uint8Array(0);
      }
      const start = Math.max(0, Number.isFinite(offset) ? Math.floor(offset) : 0);
      const size = Math.max(0, Number.isFinite(length) ? Math.floor(length) : 0);
      if (rangeCall === 3) {
        return shard.slice(start, Math.min(shard.byteLength, start + Math.max(1, size - 1)));
      }
      return shard.slice(start, Math.min(shard.byteLength, start + size));
    },
  });

  const bytes = await collectStream(cache.streamRange(0, 0, 4, { chunkBytes: 2 }));
  assert.deepEqual(bytes, [0, 1, 2, 3]);
  assert.equal(cache.lastSource?.mode, 'stream');
  assert.equal(cache.lastSource?.path, 'custom-range');
  assert.equal(cache.lastSource?.fallback, 'custom_range_partial_retry');
}

console.log('shard-cache-fallbacks.test: ok');
