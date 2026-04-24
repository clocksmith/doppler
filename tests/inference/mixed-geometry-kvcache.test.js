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

const { setDevice } = await import('../../src/gpu/device.js');
const { createKVCache } = await import('../../src/inference/pipelines/text/init.js');
const { MixedGeometryKVCache } = await import('../../src/inference/kv-cache/mixed-geometry.js');
const { DEFAULT_KVCACHE_CONFIG } = await import('../../src/config/schema/index.js');

let nextBufferId = 0;

function createMockGPUBuffer(descriptor) {
  const size = descriptor?.size ?? 0;
  return {
    label: descriptor?.label ?? '',
    size,
    usage: descriptor?.usage ?? 0,
    destroyed: false,
    _id: nextBufferId++,
    _arrayBuffer: new ArrayBuffer(size),
    destroy() {
      this.destroyed = true;
    },
    mapAsync() {
      return Promise.resolve();
    },
    getMappedRange() {
      return this._arrayBuffer;
    },
    unmap() {},
  };
}

function writeBytes(buffer, offset, data) {
  const dst = new Uint8Array(buffer._arrayBuffer, offset, data.byteLength);
  dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

function createMockDevice() {
  return {
    queue: {
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          for (const op of commandBuffer.ops ?? []) {
            const src = new Uint8Array(op.src._arrayBuffer, op.srcOffset, op.size);
            const dst = new Uint8Array(op.dst._arrayBuffer, op.dstOffset, op.size);
            dst.set(src);
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        writeBytes(buffer, offset, data);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(['shader-f16']),
    limits: {
      maxStorageBufferBindingSize: 1 << 28,
      maxBufferSize: 1 << 28,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      return {};
    },
    createBuffer(descriptor) {
      return createMockGPUBuffer(descriptor);
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ src, srcOffset, dst, dstOffset, size });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

function createSourceBuffer(device, size, fillByte) {
  const buffer = device.createBuffer({
    label: `source_${fillByte}`,
    size,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const bytes = new Uint8Array(size);
  bytes.fill(fillByte);
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

setDevice(createMockDevice(), { platformConfig: null });

const layerTypes = Array.from({ length: 35 }, (_, index) => index % 5 === 4 ? 'full_attention' : 'sliding_attention');
const modelConfig = {
  numLayers: 35,
  numKVHeads: 1,
  numGlobalKVHeads: 4,
  headDim: 256,
  globalHeadDim: 512,
  numKvSharedLayers: 20,
  maxSeqLen: 131072,
  slidingWindow: 512,
  attnLogitSoftcapping: null,
  layerTypes,
  decodeStrategy: 'incremental',
};

const runtimeKV = {
  ...DEFAULT_KVCACHE_CONFIG,
  maxSeqLen: 1024,
  kvDtype: 'f32',
  layout: 'contiguous',
  tiering: {
    ...DEFAULT_KVCACHE_CONFIG.tiering,
    mode: 'off',
  },
  quantization: {
    ...DEFAULT_KVCACHE_CONFIG.quantization,
    mode: 'none',
  },
};

{
  const cache = createKVCache(modelConfig, true, false, runtimeKV);
  assert.ok(cache instanceof MixedGeometryKVCache);
  assert.equal(cache.getGPUBuffers(0)?.layout, 'ring');
  assert.equal(cache.getGPUBuffers(4)?.layout, 'contiguous');
  assert.equal(cache.getGPUBuffers(0)?.seqLen, 0);
  assert.equal(cache.getGPUBuffers(4)?.seqLen, 0);

  const device = createMockDevice();
  setDevice(device, { platformConfig: null });
  const ringCache = createKVCache(modelConfig, true, false, runtimeKV);
  const ringSpec = ringCache.layerSpecs[0];
  const fullSpec = ringCache.layerSpecs[4];
  assert.equal(ringSpec.numHeads, 1);
  assert.equal(fullSpec.numHeads, 4);
  assert.equal(fullSpec.bytesPerToken, 4 * 512 * 4);
  const ringKeys = createSourceBuffer(device, ringSpec.bytesPerToken * 600, 7);
  const ringValues = createSourceBuffer(device, ringSpec.bytesPerToken * 600, 9);
  ringCache.updateFromGPU(0, ringKeys, ringValues, 0, 600);
  assert.equal(ringCache.getGPUBuffers(0)?.seqLen, 512);
  assert.equal(ringCache.currentSeqLen, 600);

  const fullKeys = createSourceBuffer(device, fullSpec.bytesPerToken * 4, 11);
  const fullValues = createSourceBuffer(device, fullSpec.bytesPerToken * 4, 13);
  ringCache.updateFromGPU(4, fullKeys, fullValues, 0, 4);
  assert.equal(ringCache.getGPUBuffers(4)?.seqLen, 4);

  const cloned = ringCache.clone();
  assert.ok(cloned instanceof MixedGeometryKVCache);
  assert.equal(cloned.getGPUBuffers(0)?.seqLen, 512);
  assert.equal(cloned.getGPUBuffers(4)?.seqLen, 4);
}

console.log('mixed-geometry-kvcache.test: ok');
