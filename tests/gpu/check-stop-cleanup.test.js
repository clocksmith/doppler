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

const { checkStop } = await import('../../src/gpu/kernels/check-stop.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');

class FakeBuffer {
  constructor({ size, usage, mapReject = false, data = null }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.data = data ?? new Uint32Array(Math.max(1, size / Uint32Array.BYTES_PER_ELEMENT));
    this.destroyed = false;
    this.unmapped = false;
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

function createFakeDevice({ mapReject = false } = {}) {
  const createdBuffers = [];

  const device = {
    createdBuffers,
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
      const isReadback = (usage & GPUBufferUsage.MAP_READ) !== 0;
      const buffer = new FakeBuffer({
        size,
        usage,
        mapReject: isReadback && mapReject,
      });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroupLayout(descriptor) {
      return { ...descriptor };
    },
    createPipelineLayout(descriptor) {
      return { ...descriptor };
    },
    createShaderModule(descriptor) {
      return { ...descriptor };
    },
    createComputePipeline(descriptor) {
      return { ...descriptor };
    },
    createBindGroup(descriptor) {
      return { ...descriptor };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
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
    },
  };

  return device;
}

configurePerfGuards({
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
});

{
  const device = createFakeDevice({ mapReject: true });
  setDevice(device, { platformConfig: null });

  const sampledTokenBuffer = new FakeBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
    data: new Uint32Array([7]),
  });

  await assert.rejects(
    () => checkStop({
      sampledTokenBuffer,
      eosTokenId: 7,
      maxTokens: 32,
      currentPos: 5,
    }),
    /map failed/
  );

  const [uniformBuffer, shouldStopBuffer, stagingBuffer] = device.createdBuffers;
  assert.equal(uniformBuffer.destroyed, true);
  assert.equal(shouldStopBuffer.destroyed, true);
  assert.equal(stagingBuffer.destroyed, true);
  assert.equal(stagingBuffer.unmapped, false);
}

setDevice(null);
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
console.log('check-stop-cleanup.test: ok');
