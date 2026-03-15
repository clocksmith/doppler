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

function createFailingDevice({ createBufferThrowAt = null, writeBufferThrowAt = null } = {}) {
  let createBufferCount = 0;
  let writeBufferCount = 0;
  const createdBuffers = [];

  return {
    createdBuffers,
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
        writeBufferCount += 1;
        if (writeBufferThrowAt === writeBufferCount) {
          throw new Error(`writeBuffer failed at ${writeBufferCount}`);
        }
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
      createBufferCount += 1;
      if (createBufferThrowAt === createBufferCount) {
        throw new Error(`createBuffer failed at ${createBufferCount}`);
      }
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
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

{
  const device = createFailingDevice({ createBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  const cache = new KVCache({
    numLayers: 1,
    numHeads: 1,
    headDim: 2,
    maxSeqLen: 4,
    useGPU: false,
    layout: 'contiguous',
    pageSize: 1,
    kvDtype: 'f32',
  });

  const layer = cache.layers[0];
  layer.seqLen = 1;
  layer.keys.set([1, 2]);
  layer.values.set([3, 4]);
  cache.currentSeqLen = 1;

  assert.throws(
    () => cache.setGPUContext({ device }),
    /createBuffer failed at 2/
  );

  assert.equal(cache.useGPU, false);
  assert.equal(layer.keysGPU, null);
  assert.equal(layer.valuesGPU, null);
  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].destroyed, true);
}

{
  const device = createFailingDevice({ writeBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  const cache = new KVCache({
    numLayers: 1,
    numHeads: 1,
    headDim: 2,
    maxSeqLen: 4,
    useGPU: false,
    layout: 'paged',
    pageSize: 1,
    kvDtype: 'f32',
  });

  const layer = cache.layers[0];
  layer.seqLen = 1;
  layer.keyPages[0] = new Float32Array([1, 2]);
  layer.valuePages[0] = new Float32Array([3, 4]);
  layer.allocatedPages = 1;
  cache.currentSeqLen = 1;

  assert.throws(
    () => cache.setGPUContext({ device }),
    /writeBuffer failed at 2/
  );

  assert.equal(cache.useGPU, false);
  assert.equal(layer.keysGPU, null);
  assert.equal(layer.valuesGPU, null);
  assert.equal(layer.pageTableGPU, null);
  assert.equal(device.createdBuffers.length, 3);
  for (const buffer of device.createdBuffers) {
    assert.equal(buffer.destroyed, true);
  }
}

{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const cache = new KVCache({
    numLayers: 4,
    numHeads: 1,
    headDim: 2,
    maxSeqLen: 8,
    useGPU: true,
    layout: 'contiguous',
    pageSize: 1,
    kvDtype: 'f32',
  });

  const bytesPerToken = cache.kvSize * Float32Array.BYTES_PER_ELEMENT;
  const sourceK = new FakeBuffer({
    size: 2 * bytesPerToken,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
  });
  const sourceV = new FakeBuffer({
    size: 2 * bytesPerToken,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
  });

  cache.updateFromGPU(1, sourceK, sourceV, 0, 2);

  assert.equal(
    cache.currentSeqLen,
    2,
    'Hybrid cache updates must advance global seqLen even when attention is not the last layer'
  );
  assert.equal(cache.totalTokensSeen, 2);
  assert.equal(cache.layers[1].seqLen, 2);
  assert.equal(cache.getMemoryStats().seqLen, 2);
  assert.equal(cache.getMemoryStats().used, 4 * 2 * 2 * cache.kvSize * Float32Array.BYTES_PER_ELEMENT);

  cache.destroy();
}

destroyBufferPool();
setDevice(null);
console.log('kv-cache-sync.test: ok');
