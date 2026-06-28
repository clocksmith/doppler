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

const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { FEATURES, setDevice } = await import('../../src/gpu/device.js');
const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { SlidingWindowKVCache } = await import('../../src/inference/kv-cache/sliding-window.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { resetUniformCache } = await import('../../src/gpu/uniform-cache.js');
const { clearKernelCaches } = await import('../../src/gpu/kernels/index.js');

class FakeBuffer {
  constructor({ label = null, size, usage }) {
    this.label = label;
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

function createFakeDevice() {
  const createdBuffers = [];
  const computePasses = [];
  const copyOps = [];

  return {
    createdBuffers,
    computePasses,
    copyOps,
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set([FEATURES.SHADER_F16]),
    limits: {
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
      maxQuerySetSize: 64,
    },
    createBuffer({ label, size, usage }) {
      const buffer = new FakeBuffer({ label, size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      return {};
    },
    createBindGroupLayout() {
      return {};
    },
    createPipelineLayout() {
      return {};
    },
    createShaderModule() {
      return {
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    async createComputePipelineAsync() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass(descriptor = {}) {
          const passRecord = {
            label: descriptor.label ?? null,
            ended: false,
            dispatches: [],
          };
          computePasses.push(passRecord);
          return {
            setPipeline(pipeline) {
              passRecord.pipeline = pipeline;
            },
            setBindGroup(index, bindGroup) {
              passRecord.bindGroup = { index, bindGroup };
            },
            dispatchWorkgroups(x, y, z) {
              passRecord.dispatches.push({ x, y, z });
            },
            end() {
              passRecord.ended = true;
            },
          };
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          copyOps.push({ source, sourceOffset, destination, destinationOffset, size });
        },
        finish() {
          return {};
        },
      };
    },
    destroy() {},
  };
}

function resetRuntime(device = null) {
  destroyBufferPool();
  resetUniformCache();
  clearKernelCaches();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function createSourceBuffer(elementCount) {
  return new FakeBuffer({
    size: elementCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

try {
  {
    const device = createFakeDevice();
    resetRuntime(device);
    const cache = new KVCache({
      numLayers: 1,
      numHeads: 2,
      headDim: 4,
      maxSeqLen: 8,
      useGPU: true,
      layout: 'contiguous',
      pageSize: 8,
      kvDtype: 'f16',
    });
    const recorder = new CommandRecorder(device, 'kv_direct_contiguous');
    const sourceElements = 2 * cache.kvSize;

    await cache.recordUpdateF32ToF16FromGPU(
      recorder,
      0,
      createSourceBuffer(sourceElements),
      createSourceBuffer(sourceElements),
      2,
      2
    );

    assert.equal(device.copyOps.length, 0);
    assert.equal(recorder.getStats().opCount, 1);
    assert.equal(recorder.getStats().computePassCount, 1);
    assert.equal(recorder.getStats().opLabelCounts['kv_cache_write:f32_to_f16'], 1);
    assert.equal(cache.currentSeqLen, 4);
    assert.equal(cache.layers[0].seqLen, 4);

    recorder.abort();
    cache.destroy();
    resetRuntime();
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const cache = new KVCache({
      numLayers: 1,
      numHeads: 2,
      headDim: 4,
      maxSeqLen: 8,
      useGPU: true,
      layout: 'contiguous',
      pageSize: 8,
      kvDtype: 'f16',
    });

    cache.recordF16UpdateAlreadyWrittenFromGPU(0, 3, 2);

    assert.equal(device.copyOps.length, 0);
    assert.equal(device.computePasses.length, 0);
    assert.equal(cache.counters.recordedGpuUpdateCalls, 1);
    assert.equal(cache.counters.tokensWritten, 2);
    assert.equal(cache.currentSeqLen, 5);
    assert.equal(cache.totalTokensSeen, 5);
    assert.equal(cache.layers[0].seqLen, 5);
    assert.throws(
      () => cache.recordF16UpdateAlreadyWrittenFromGPU(0, 7, 2),
      /Cache overflow/
    );

    cache.destroy();
    resetRuntime();
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const cache = new SlidingWindowKVCache({
      numLayers: 1,
      numHeads: 1,
      headDim: 4,
      maxSeqLen: 8,
      windowSize: 4,
      useGPU: true,
      layout: 'contiguous',
      pageSize: 4,
      kvDtype: 'f16',
    });
    const recorder = new CommandRecorder(device, 'kv_direct_sliding_wrap');
    const sourceElements = 3 * cache.kvSize;

    await cache.recordUpdateF32ToF16FromGPU(
      recorder,
      0,
      createSourceBuffer(sourceElements),
      createSourceBuffer(sourceElements),
      3,
      3
    );

    assert.equal(device.copyOps.length, 0);
    assert.equal(recorder.getStats().opCount, 2);
    assert.equal(recorder.getStats().computePassCount, 1);
    assert.equal(recorder.getStats().opLabelCounts['kv_cache_write:f32_to_f16'], 2);
    assert.equal(cache.currentSeqLen, 4);
    assert.equal(cache.layers[0].seqLen, 4);

    recorder.abort();
    cache.destroy();
    resetRuntime();
  }
} finally {
  resetRuntime();
  if (ORIGINAL_GPU_BUFFER === undefined) {
    delete globalThis.GPUBuffer;
  } else {
    globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
  }
}

console.log('kv-cache-direct-write.test: ok');
