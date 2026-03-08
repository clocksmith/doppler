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

globalThis.GPUShaderStage = {
  COMPUTE: 0x2,
};

const {
  destroyBufferPool,
  getBufferPool,
} = await import('../../src/memory/buffer-pool.js');
const { PartitionedBufferPool } = await import('../../src/gpu/partitioned-buffer-pool.js');
const {
  getSubmitStats,
  resetSubmitStats,
  setTrackSubmits,
  wrapQueueForTracking,
} = await import('../../src/gpu/submit-tracker.js');
const {
  generateCacheKey,
  getDeviceSignature,
  loadCache,
} = await import('../../src/gpu/kernel-tuner/cache.js');
const { tuneSoftmax } = await import('../../src/gpu/kernel-tuner/benchmarks.js');
const { getKernelTuner } = await import('../../src/gpu/kernel-tuner/tuner.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { getRuntimeConfig, setRuntimeConfig } = await import('../../src/config/runtime.js');
const { getPerfConfig } = await import('../../src/gpu/perf-guards.js');
const { UniformBufferCache } = await import('../../src/gpu/uniform-cache.js');
const { DEFAULT_PERF_GUARDS_CONFIG } = await import('../../src/config/schema/debug.schema.js');

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

function createBaseDevice(overrides = {}) {
  const createdBuffers = [];
  const queue = overrides.queue ?? {
    submit() {},
    writeBuffer() {},
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };
  return {
    createdBuffers,
    queue,
    lost: overrides.lost ?? new Promise(() => {}),
    features: overrides.features ?? new Set(),
    limits: overrides.limits ?? {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup: overrides.createBindGroup ?? function createBindGroup(descriptor) {
      return descriptor;
    },
    createBindGroupLayout: overrides.createBindGroupLayout ?? function createBindGroupLayout(descriptor) {
      return descriptor;
    },
    createPipelineLayout: overrides.createPipelineLayout ?? function createPipelineLayout(descriptor) {
      return descriptor;
    },
    createShaderModule: overrides.createShaderModule ?? function createShaderModule() {
      return {};
    },
    createComputePipelineAsync: overrides.createComputePipelineAsync ?? async function createComputePipelineAsync() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createCommandEncoder: overrides.createCommandEncoder ?? function createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() {
          return {};
        },
      };
    },
    destroy: overrides.destroy ?? function destroy() {},
  };
}

function createCacheStorage() {
  const store = new Map();
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
    removeItem(key) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
  };
}

const originalRuntimeConfig = getRuntimeConfig();
const originalLocalStorage = globalThis.localStorage;

setRuntimeConfig({
  shared: {
    debug: {
      profiler: {
        queryCapacity: 4,
        maxSamples: 4,
        maxDurationMs: 1000,
      },
    },
    tuner: {
      cacheKeyPrefix: 'test_tuner_',
      defaultWarmupIterations: 0,
      defaultTimedIterations: 1,
    },
    bufferPool: {
      enablePooling: false,
      maxPoolSizePerBucket: 8,
      maxTotalPooledBuffers: 16,
      alignmentBytes: 16,
    },
  },
});

try {
  {
    destroyBufferPool();
    const device = createBaseDevice();
    setDevice(device, { platformConfig: null });
    getBufferPool().configure({ enablePooling: false });

    const pool = new PartitionedBufferPool([{ id: 'expertA' }, { id: 'expertB' }]);
    const buffer = pool.acquire('expertA', 64, GPUBufferUsage.STORAGE, 'partitioned');

    assert.equal(pool.getExpertPool('expertA')?.isActiveBuffer(buffer), true);
    pool.release('expertB', buffer);
    assert.equal(pool.getExpertPool('expertA')?.isActiveBuffer(buffer), false);
    assert.equal(pool.getExpertPool('expertB')?.isActiveBuffer(buffer), false);
  }

  {
    resetSubmitStats();
    setTrackSubmits(true);
    let submitCalls = 0;
    const queue = {
      submit() {
        submitCalls += 1;
      },
    };

    wrapQueueForTracking(queue);
    wrapQueueForTracking(queue);
    queue.submit([]);

    const stats = getSubmitStats();
    assert.equal(submitCalls, 1);
    assert.equal(stats.count, 1);
    setTrackSubmits(false);
  }

  {
    const storage = createCacheStorage();
    globalThis.localStorage = storage;

    const fallbackSignature = getDeviceSignature({
      adapterInfo: { vendor: 'unknown', architecture: 'unknown', device: 'unknown' },
      hasF16: true,
      hasSubgroups: false,
      hasSubgroupsF16: false,
      hasTimestampQuery: false,
      maxBufferSize: 4096,
      maxWorkgroupSize: 128,
      maxWorkgroupStorageSize: 8192,
    });
    assert.match(fallbackSignature, /f16/);
    assert.doesNotMatch(fallbackSignature, /^unknown_unknown_unknown$/);

    const invalidCacheKey = 'test_tuner_' + fallbackSignature;
    storage.setItem(invalidCacheKey, JSON.stringify({
      bad_record: {
        optimalWorkgroupSize: [64, 1, 1],
        optimalTileSize: 64,
        throughput: 1,
        timeMs: 'bad',
      },
    }));

    const loaded = loadCache({
      adapterInfo: { vendor: 'unknown', architecture: 'unknown', device: 'unknown' },
      hasF16: true,
      hasSubgroups: false,
      hasSubgroupsF16: false,
      hasTimestampQuery: false,
      maxBufferSize: 4096,
      maxWorkgroupSize: 128,
      maxWorkgroupStorageSize: 8192,
    });
    assert.equal(loaded.size, 0);
    assert.equal(storage.getItem(invalidCacheKey), null);
  }

  {
    const storage = createCacheStorage();
    globalThis.localStorage = storage;

    const cacheKey = generateCacheKey('softmax', { innerSize: 8, outerSize: 1 });
    storage.setItem('test_tuner_vendor_arch_devA', JSON.stringify({
      [cacheKey]: {
        optimalWorkgroupSize: [64, 1, 1],
        optimalTileSize: 64,
        throughput: 10,
        timeMs: 4,
        deviceInfo: { vendor: 'vendor', architecture: 'arch', device: 'devA' },
      },
    }));
    storage.setItem('test_tuner_vendor_arch_devB', JSON.stringify({
      [cacheKey]: {
        optimalWorkgroupSize: [128, 1, 1],
        optimalTileSize: 128,
        throughput: 20,
        timeMs: 2,
        deviceInfo: { vendor: 'vendor', architecture: 'arch', device: 'devB' },
      },
    }));

    setDevice(createBaseDevice(), {
      platformConfig: null,
      adapterInfo: { vendor: 'vendor', architecture: 'arch', device: 'devA' },
    });
    const tunerA = await getKernelTuner();
    assert.deepEqual(tunerA.getCachedResult('softmax', { innerSize: 8, outerSize: 1 })?.optimalWorkgroupSize, [64, 1, 1]);

    setDevice(createBaseDevice(), {
      platformConfig: null,
      adapterInfo: { vendor: 'vendor', architecture: 'arch', device: 'devB' },
    });
    const tunerB = await getKernelTuner();
    assert.equal(tunerA, tunerB);
    assert.deepEqual(tunerB.getCachedResult('softmax', { innerSize: 8, outerSize: 1 })?.optimalWorkgroupSize, [128, 1, 1]);
  }

  {
    const device = createBaseDevice({
      createBindGroup() {
        throw new Error('createBindGroup failed');
      },
    });

    const result = await tuneSoftmax(
      device,
      { innerSize: 16, outerSize: 1 },
      [[64, 1, 1]],
      0,
      1,
      { adapterInfo: { vendor: 'vendor', architecture: 'arch', device: 'dev' } }
    );

    assert.equal(result.optimalTileSize, 256);
    for (const buffer of device.createdBuffers) {
      assert.equal(buffer.destroyed, true);
    }
  }

  {
    const device = createBaseDevice();
    setDevice(device, { platformConfig: null });
    const cache = new UniformBufferCache(8, 1000);
    const first = cache.getOrCreate(new Uint8Array([0, 0, 0, 1]).buffer, 'u0');
    cache.release(first);
    const second = cache.getOrCreate(new Uint8Array([0, 0, 0, 2]).buffer, 'u1');

    assert.notEqual(first, second);
    assert.equal(cache.getStats().misses, 2);
  }

  {
    assert.deepEqual(getPerfConfig(), DEFAULT_PERF_GUARDS_CONFIG);
  }
} finally {
  globalThis.localStorage = originalLocalStorage;
  setRuntimeConfig(originalRuntimeConfig);
  destroyBufferPool();
  setDevice(null);
}

console.log('runtime-infra-regressions.test: ok');
