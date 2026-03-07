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

const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { destroyBufferPool } = await import('../../src/memory/buffer-pool.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.unmapped = false;
    this.data = new Uint8Array(size);
  }

  async mapAsync() {}

  getMappedRange(offset = 0, size = this.size - offset) {
    return this.data.slice(offset, offset + size).buffer;
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function copyBytes(src, srcOffset, dst, dstOffset, size) {
  const srcBytes = src.data.subarray(srcOffset, srcOffset + size);
  dst.data.set(srcBytes, dstOffset);
}

function createFakeDevice() {
  return {
    queue: {
      submit(commandBuffers) {
        for (const buffer of commandBuffers) {
          for (const op of buffer.ops) {
            if (op.type === 'copyBufferToBuffer') {
              copyBytes(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
            }
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        buffer.data.set(bytes, offset);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
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
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      return {};
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ type: 'copyBufferToBuffer', src, srcOffset, dst, dstOffset, size });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

configurePerfGuards({
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
});

{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const cache = new KVCache({
    numLayers: 1,
    numHeads: 1,
    headDim: 2,
    maxSeqLen: 4,
    useGPU: true,
    layout: 'contiguous',
    pageSize: 1,
    kvDtype: 'f32',
  });

  const layer = cache.layers[0];
  layer.seqLen = 2;
  new Float32Array(layer.keysGPU.data.buffer).set([1, 2, 3, 4]);
  new Float32Array(layer.valuesGPU.data.buffer).set([5, 6, 7, 8]);

  await cache.syncToCPU();

  assert.deepEqual(Array.from(layer.keys.slice(0, 4)), [1, 2, 3, 4]);
  assert.deepEqual(Array.from(layer.values.slice(0, 4)), [5, 6, 7, 8]);

  cache.destroy();
}

destroyBufferPool();
setDevice(null);
console.log('kv-cache-sync.test: ok');
