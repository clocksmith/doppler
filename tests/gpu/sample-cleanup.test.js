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

globalThis.GPUMapMode = {
  READ: 1 << 0,
  WRITE: 1 << 1,
};

globalThis.GPUShaderStage = {
  COMPUTE: 0x2,
};

const { runArgmax, runGPUSample } = await import('../../src/gpu/kernels/sample.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');

class FakeBuffer {
  constructor({ size, usage, mapReject = false, mappedValue = 0 }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.mappedValue = mappedValue;
    this.destroyed = false;
    this.unmapped = false;
    this.data = new Uint32Array(Math.max(1, size / Uint32Array.BYTES_PER_ELEMENT));
    this.data[0] = mappedValue;
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange() {
    return this.data.buffer;
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

function createFakePipeline() {
  return {
    getBindGroupLayout() {
      return { label: 'layout' };
    },
  };
}

function createFakeDevice({ mapReject = false } = {}) {
  const createdBuffers = [];
  const encoders = [];

  return {
    createdBuffers,
    encoders,
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      writeBuffer(buffer, offset, source) {
        const bytes = source instanceof ArrayBuffer
          ? new Uint8Array(source)
          : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
        const target = new Uint8Array(buffer.data.buffer);
        target.set(bytes, offset);
      },
    },
    features: new Set(),
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
    },
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({
        size,
        usage,
        mapReject: (usage & GPUBufferUsage.MAP_READ) !== 0 && mapReject,
      });
      createdBuffers.push(buffer);
      return buffer;
    },
    createShaderModule({ code, label }) {
      return {
        code,
        label,
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    createComputePipeline() {
      return createFakePipeline();
    },
    async createComputePipelineAsync() {
      return createFakePipeline();
    },
    createBindGroupLayout(descriptor) {
      return { ...descriptor };
    },
    createPipelineLayout(descriptor) {
      return { ...descriptor };
    },
    createBindGroup(descriptor) {
      return { ...descriptor };
    },
    createCommandEncoder() {
      const encoder = {
        passes: [],
        beginComputePass() {
          const pass = { label: null };
          encoder.passes.push(pass);
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
      encoders.push(encoder);
      return encoder;
    },
  };
}

function resetRuntimeState(device) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: false,
  });
}

{
  const device = createFakeDevice({ mapReject: true });
  resetRuntimeState(device);

  const logits = new FakeBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });

  await assert.rejects(
    () => runArgmax(logits, 32, {
      padTokenId: null,
      logitSoftcap: 0,
      logitsDtype: 'f32',
      outputIndex: 0,
    }),
    /map failed/
  );

  const poolStats = getBufferPool().getStats();
  const stagingBuffer = device.createdBuffers.at(-1);
  assert.equal(stagingBuffer.destroyed, true);
  assert.equal(poolStats.activeBuffers, 0);
  assert.equal(poolStats.currentBytesAllocated, 0);
}

{
  const device = createFakeDevice({ mapReject: true });
  resetRuntimeState(device);

  const logits = new FakeBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });

  await assert.rejects(
    () => runGPUSample(logits, 32, {
      temperature: 1,
      topK: 4,
      greedyThreshold: 0.1,
      randomSeed: 123,
      padTokenId: null,
      logitSoftcap: 0,
      logitsDtype: 'f32',
      outputIndex: 0,
    }),
    /map failed/
  );

  const poolStats = getBufferPool().getStats();
  const stagingBuffer = device.createdBuffers.at(-1);
  assert.equal(stagingBuffer.destroyed, true);
  assert.equal(poolStats.activeBuffers, 0);
  assert.equal(poolStats.currentBytesAllocated, 0);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const logits = new FakeBuffer({
    size: 65536 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });

  await runArgmax(logits, 65536, {
    padTokenId: null,
    logitSoftcap: 0,
    logitsDtype: 'f32',
    outputIndex: 0,
  });

  const totalPasses = device.encoders.reduce((count, encoder) => count + encoder.passes.length, 0);
  assert.equal(totalPasses, 2);
}

destroyBufferPool();
setDevice(null);
console.log('sample-cleanup.test: ok');
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
