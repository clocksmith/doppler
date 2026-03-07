import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const { ShardCache } = await import('../../src/loader/shard-cache.js');
const { ExpertCache } = await import('../../src/loader/experts/expert-cache.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { acquireBuffer, destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { DopplerLoader } = await import('../../src/loader/doppler-loader.js');
const { clearManifest, getExpertBytes, getShardsForExpert } = await import('../../src/formats/rdrr/index.js');

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
    modelId: 'loader-runtime-cleanup-test',
    hashAlgorithm: 'sha256',
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size,
        hash: '0'.repeat(64),
        hashAlgorithm: 'sha256',
        offset: 0,
      },
    ],
  };
}

function createMoeManifest() {
  return {
    modelId: 'loader-runtime-cleanup-moe',
    hashAlgorithm: 'sha256',
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size: 64,
        hash: '0'.repeat(64),
        hashAlgorithm: 'sha256',
        offset: 0,
      },
    ],
    moeConfig: {
      numExperts: 2,
      numExpertsPerToken: 1,
    },
    groups: {
      'layer.0.expert.0': {
        shards: [0],
        tensors: ['layers.0.block_sparse_moe.experts.0.w1.weight'],
      },
    },
  };
}

async function collectStream(iterable) {
  const out = [];
  for await (const chunk of iterable) {
    out.push(...chunk);
  }
  return out;
}

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  return {
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
  };
}

function resetBufferRuntime() {
  destroyBufferPool();
  setDevice(null);
}

{
  const cache = new ShardCache({
    maxEntries: 2,
    loadingConfig: createLoadingConfig(),
    verifyHashes: false,
    manifest: createManifest(4),
    customRangeLoader: async (_index, offset, length) => {
      const start = Math.max(0, Number.isFinite(offset) ? Math.floor(offset) : 0);
      const size = Math.max(0, Number.isFinite(length) ? Math.floor(length) : 0);
      if (start === 0) {
        return new Uint8Array([0, 1].slice(0, size));
      }
      return new Uint8Array(0);
    },
  });

  await assert.rejects(
    () => collectStream(cache.streamRange(0, 0, 4, { chunkBytes: 2 })),
    /short stream read/
  );
}

{
  let releaseFirstLoad;
  const firstLoadReleased = new Promise((resolve) => {
    releaseFirstLoad = resolve;
  });
  let signalFirstStart;
  const firstStarted = new Promise((resolve) => {
    signalFirstStart = resolve;
  });
  let callCount = 0;
  const cache = new ShardCache({
    maxEntries: 2,
    loadingConfig: createLoadingConfig(),
    verifyHashes: false,
    manifest: createManifest(4),
    customLoader: async () => {
      callCount += 1;
      if (callCount === 1) {
        signalFirstStart();
        await firstLoadReleased;
        return new Uint8Array([1, 1, 1, 1]);
      }
      return new Uint8Array([2, 2, 2, 2]);
    },
  });

  const stalePrefetch = cache.prefetch(0);
  await firstStarted;
  cache.clear();
  const freshLoad = cache.load(0);
  releaseFirstLoad();

  const staleBytes = new Uint8Array(await stalePrefetch);
  const freshBytes = new Uint8Array(await freshLoad);
  assert.equal(callCount, 2);
  assert.deepEqual(Array.from(staleBytes), [1, 1, 1, 1]);
  assert.deepEqual(Array.from(freshBytes), [2, 2, 2, 2]);
  assert.deepEqual(Array.from(new Uint8Array(await cache.load(0))), [2, 2, 2, 2]);
}

{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });

  const cache = new ExpertCache();
  const gateBuffer = acquireBuffer(16, GPUBufferUsage.STORAGE, 'expert_gate');
  const weights = {
    gate: createWeightBuffer(gateBuffer, 'f16', 'row', [1, 8], 'expert_gate'),
    up: null,
    down: null,
  };

  cache.put(0, 0, weights, 16);
  cache.clear();
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(gateBuffer.destroyed, true);
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
  resetBufferRuntime();
}

{
  clearManifest();
  const manifest = createMoeManifest();
  const loader = new DopplerLoader();
  loader.setManifest(manifest);

  assert.deepEqual(getShardsForExpert(0, 0), [0]);
  assert.equal(getExpertBytes(), 64);
  await loader.unload();

  assert.deepEqual(getShardsForExpert(0, 0, manifest), [0]);
  assert.throws(() => getShardsForExpert(0, 0), /Missing expert group mapping/);
}

console.log('loader-runtime-cleanup.test: ok');
