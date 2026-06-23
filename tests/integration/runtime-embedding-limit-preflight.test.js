import assert from 'node:assert/strict';
import fs from 'node:fs';

import { createDopplerRuntimeService } from '../../src/client/runtime/index.js';
import { setDevice } from '../../src/gpu/device.js';
import { destroyBufferPool } from '../../src/memory/buffer-pool.js';
import { resolveManifestGpuResidentEmbeddingLimitError } from '../../src/loader/embedding-limit-preflight.js';

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

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const embeddingName = 'model.language_model.embed_tokens.weight';
const manifest = {
  modelId: 'gemma-4-e2b-preflight-test',
  inference: {
    largeWeights: {
      gpuResidentOverrides: [embeddingName],
    },
    execution: {
      kernels: {
        embed: {
          kernel: 'gather.wgsl',
          entry: 'main',
        },
      },
    },
  },
  tensors: {
    [embeddingName]: {
      shape: [128, 2],
      dtype: 'F16',
      role: 'embedding',
      group: 'embed',
      layout: 'row',
    },
  },
};
const split8Manifest = {
  ...manifest,
  inference: {
    ...manifest.inference,
    execution: {
      kernels: {
        embed: {
          kernel: 'gather_split8_f16_vec4_f32_out.wgsl',
          entry: 'gather_vec4_f32_out',
        },
      },
    },
  },
};

try {
  destroyBufferPool();
  setDevice(createTinyStorageLimitDevice(), { platformConfig: null });

  assert.equal(
    resolveManifestGpuResidentEmbeddingLimitError(manifest, {
      runtimeConfig: {
        inference: {
          largeWeights: {
            gpuResidentOverrides: [],
          },
        },
      },
    }),
    null,
    'empty runtime override must clear manifest GPU-resident embedding policy'
  );

  setDevice(createTinyStorageLimitDevice({ maxStorageBuffersPerShaderStage: 10 }), { platformConfig: null });
  assert.equal(
    resolveManifestGpuResidentEmbeddingLimitError(split8Manifest),
    null,
    'split8 manifest should pass preflight when the adapter can bind the split8 shader'
  );

  setDevice(createTinyStorageLimitDevice({
    maxStorageBufferBindingSize: 134217728,
    maxBufferSize: 4294967295,
    maxStorageBuffersPerShaderStage: 10,
  }), { platformConfig: null });
  for (const filePath of [
    'models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json',
    'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json',
    'models/local/gemma-4-e2b-it-q4k-ehf16-af16-int4ple/manifest.json',
  ]) {
    assert.equal(
      resolveManifestGpuResidentEmbeddingLimitError(readJson(filePath)),
      null,
      `${filePath} should clear GPU-resident embedding preflight on a split8-capable adapter`
    );
  }

  setDevice(createTinyStorageLimitDevice({ maxStorageBuffersPerShaderStage: 8 }), { platformConfig: null });
  const split8LimitError = resolveManifestGpuResidentEmbeddingLimitError(split8Manifest);
  assert.ok(split8LimitError, 'split8 manifest should fail preflight when storage-buffer bindings are too low');
  assert.equal(
    split8LimitError.details?.weightLoadFailure?.deviceLimitFailure?.splitKernelExpected,
    true
  );
  assert.equal(
    split8LimitError.details?.weightLoadFailure?.deviceLimitFailure?.activeSplitKernelMaxSections,
    8
  );
  assert.equal(
    split8LimitError.details?.weightLoadFailure?.deviceLimitFailure?.maxSplitEmbeddingSections,
    0
  );

  const doppler = createDopplerRuntimeService({
    async ensureWebGPUAvailable() {
      setDevice(createTinyStorageLimitDevice(), { platformConfig: null });
    },
  });

  await assert.rejects(
    () => doppler.load({
      manifest,
      baseUrl: 'http://127.0.0.1:1/unreachable-model/',
    }),
    (error) => {
      assert.match(error.message, /cannot be GPU-resident on this device/);
      assert.equal(
        error.details?.weightLoadFailure?.tensorLoadStage,
        'gpuResidentEmbeddingLimitPreflight'
      );
      assert.equal(
        error.details?.weightLoadFailure?.deviceLimitFailure?.kind,
        'gpu_resident_embedding_exceeds_device_limit'
      );
      return true;
    }
  );
} finally {
  destroyBufferPool();
  setDevice(null, { platformConfig: null });
}

console.log('runtime-embedding-limit-preflight.test: ok');
