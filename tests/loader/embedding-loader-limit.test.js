import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { setDevice } from '../../src/gpu/device.js';
import { loadEmbeddings } from '../../src/loader/embedding-loader.js';
import { destroyBufferPool, getBufferPool } from '../../src/memory/buffer-pool.js';

globalThis.GPUBufferUsage ??= {
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

class FakeBuffer {
  constructor({ size, usage, label = '' }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createTinyStorageLimitDevice({
  maxStorageBufferBindingSize = 128,
  maxBufferSize = 128,
  maxStorageBuffersPerShaderStage = 8,
} = {}) {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize,
      maxBufferSize,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
      minStorageBufferOffsetAlignment: 16,
    },
    createBuffer({ size, usage, label }) {
      return new FakeBuffer({ size, usage, label });
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      return {
        finish() {
          return {};
        },
      };
    },
  };
}

const embeddingName = 'model.language_model.embed_tokens.weight';
let loadTensorCalls = 0;

try {
  destroyBufferPool();
  setDevice(createTinyStorageLimitDevice(), { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });
  setRuntimeConfig({
    inference: {
      largeWeights: {
        enabled: true,
        safetyRatio: 1,
        gpuResidentOverrides: [embeddingName],
      },
    },
  });

  await assert.rejects(
    () => loadEmbeddings({
      tensorLocations: new Map([
        [embeddingName, {
          shape: [128, 2],
          dtype: 'F16',
          role: 'embedding',
          group: 'embed',
          layout: 'row',
        }],
      ]),
      gpuBuffers: new Set(),
      hostHasShaderF16: true,
      embeddingKernel: {
        kernel: 'gather.wgsl',
        entry: 'main',
      },
      loadShardRange() {
        throw new Error('loadShardRange should not be called');
      },
      async loadTensor() {
        loadTensorCalls += 1;
        throw new Error('loadTensor should not be called');
      },
      shouldStreamLargeWeight() {
        return false;
      },
      resolveWeightLayout(location) {
        return location.layout ?? 'row';
      },
    }),
    (error) => {
      assert.match(error.message, /cannot be GPU-resident on this device/);
      const failure = error.details?.weightLoadFailure;
      assert.equal(failure?.tensorName, embeddingName);
      assert.equal(failure?.tensorRole, 'embedding');
      assert.equal(failure?.tensorDtype, 'F16');
      assert.deepEqual(failure?.tensorShape, [128, 2]);
      assert.equal(failure?.tensorSizeBytes, 512);
      assert.equal(failure?.tensorLoadStage, 'gpuResidentEmbeddingLimitPreflight');
      assert.equal(failure?.toGPU, true);
      assert.equal(failure?.streamedUpload, false);
      assert.equal(
        failure?.deviceLimitFailure?.kind,
        'gpu_resident_embedding_exceeds_device_limit'
      );
      assert.equal(failure?.deviceLimitFailure?.maxGpuResidentBytes, 128);
      assert.equal(failure?.deviceLimitFailure?.maxStorageBufferBindingSize, 128);
      assert.equal(failure?.deviceLimitFailure?.maxBufferSize, 128);
      assert.equal(failure?.deviceLimitFailure?.maxStorageBuffersPerShaderStage, 8);
      assert.equal(failure?.deviceLimitFailure?.largeWeightMaxBytes, 128);
      assert.deepEqual(failure?.deviceLimitFailure?.embeddingKernel, {
        kernel: 'gather.wgsl',
        entry: 'main',
      });
      assert.equal(failure?.deviceLimitFailure?.splitKernelExpected, false);
      assert.equal(failure?.deviceLimitFailure?.activeSplitKernelMaxSections, null);
      assert.equal(failure?.deviceLimitFailure?.maxSplitEmbeddingSections, 6);
      assert.equal(failure?.deviceLimitFailure?.requiredSplitSections, 4);
      return true;
    }
  );

  assert.equal(loadTensorCalls, 0);

  destroyBufferPool();
  setDevice(createTinyStorageLimitDevice({
    maxStorageBufferBindingSize: 64,
    maxBufferSize: 64,
    maxStorageBuffersPerShaderStage: 10,
  }), { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });
  loadTensorCalls = 0;
  let rangeReads = 0;
  const gpuBuffers = new Set();

  const splitTensor = await loadEmbeddings({
    tensorLocations: new Map([
      [embeddingName, {
        shape: [14, 16],
        dtype: 'F16',
        role: 'embedding',
        group: 'embed',
        layout: 'row',
        shardIndex: 0,
        offset: 0,
        size: 448,
      }],
    ]),
    gpuBuffers,
    hostHasShaderF16: true,
    embeddingKernel: {
      kernel: 'gather_split8_f16_vec4_f32_out.wgsl',
      entry: 'gather_vec4_f32_out',
    },
    async loadShardRange(shardIndex, offset, size) {
      assert.equal(shardIndex, 0);
      assert.equal(offset, rangeReads * 64);
      assert.equal(size, 64);
      rangeReads += 1;
      return new Uint8Array(size).buffer;
    },
    async loadTensor() {
      loadTensorCalls += 1;
      throw new Error('loadTensor should not be called for split8 embedding');
    },
    shouldStreamLargeWeight() {
      return false;
    },
    resolveWeightLayout(location) {
      return location.layout ?? 'row';
    },
  });

  assert.equal(splitTensor?.kind, 'split_weight_buffer');
  assert.equal(splitTensor.sections.length, 7);
  assert.equal(splitTensor.metadata?.splitGatherSectionCount, 8);
  assert.deepEqual(splitTensor.metadata?.sourceKernel, {
    kernel: 'gather_split8_f16_vec4_f32_out.wgsl',
    entry: 'gather_vec4_f32_out',
  });
  assert.deepEqual(
    splitTensor.sections.map((section) => [section.rowStart, section.rowCount]),
    [
      [0, 2],
      [2, 2],
      [4, 2],
      [6, 2],
      [8, 2],
      [10, 2],
      [12, 2],
    ]
  );
  assert.equal(rangeReads, 7);
  assert.equal(loadTensorCalls, 0);
  assert.equal(gpuBuffers.size, 7);
} finally {
  resetRuntimeConfig();
  destroyBufferPool();
  setDevice(null, { platformConfig: null });
}

console.log('embedding-loader-limit.test: ok');
