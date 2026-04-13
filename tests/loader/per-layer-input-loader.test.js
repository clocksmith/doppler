import assert from 'node:assert/strict';

import { loadPerLayerInputWeights } from '../../src/loader/per-layer-input-loader.js';
import { isCpuWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { setDevice } from '../../src/gpu/device.js';
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
    this._arrayBuffer = new ArrayBuffer(size);
  }

  destroy() {
    this.destroyed = true;
  }
}

function writeBytes(targetBuffer, offset, data) {
  const source = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  new Uint8Array(targetBuffer._arrayBuffer, offset, source.byteLength).set(source);
}

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer(buffer, offset, data) {
        writeBytes(buffer, offset, data);
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

destroyBufferPool();
setDevice(createFakeDevice(), { platformConfig: null });
getBufferPool().configure({ enablePooling: false });

const embedName = 'model.language_model.embed_tokens_per_layer.weight';
const projectionName = 'model.language_model.per_layer_model_projection.weight';
const projectionNormName = 'model.language_model.per_layer_projection_norm.weight';

const tensorLocations = new Map([
  [embedName, {
    shape: [16, 32],
    dtype: 'BF16',
    role: 'embedding',
    layout: 'row',
  }],
  [projectionName, {
    shape: [32, 64],
    dtype: 'F16',
    role: 'matmul',
    layout: 'row',
  }],
  [projectionNormName, {
    shape: [32],
    dtype: 'F32',
    role: 'norm',
    layout: 'row',
  }],
]);

const loadCalls = [];
const weights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-test',
  perLayerInputSession: {
    materialization: 'cpu_resident',
  },
  tensorLocations,
  async loadTensor(name, toGPU) {
    loadCalls.push([name, toGPU]);
    if (name === embedName) {
      return new Float32Array(16 * 32);
    }
    if (name === projectionName) {
      return new Float32Array(32 * 64);
    }
    if (name === projectionNormName) {
      return new Float32Array(32);
    }
    return null;
  },
  shouldStreamLargeWeight(name) {
    return name === embedName;
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 32,
});

assert.ok(weights, 'per-layer input weights should load');
assert.deepEqual(
  [...loadCalls].sort(([nameA], [nameB]) => nameA.localeCompare(nameB)),
  [
    [embedName, false],
    [projectionName, true],
    [projectionNormName, true],
  ].sort(([nameA], [nameB]) => nameA.localeCompare(nameB))
);
assert.ok(isCpuWeightBuffer(weights.embedTokensPerLayer), 'streamed embed tensor should be wrapped as CpuWeightBuffer');
assert.equal(weights.embedTokensPerLayer.dtype, 'f32');
assert.equal(weights.embedTokensPerLayer.layout, 'row');

const rangeLocation = {
  shape: [4, 4],
  dtype: 'F16',
  role: 'embedding',
  layout: 'row',
  size: 32,
  spans: [
    { shardIndex: 0, offset: 0, size: 16 },
    { shardIndex: 1, offset: 0, size: 16 },
  ],
};
const rangeTensorLocations = new Map([
  [embedName, rangeLocation],
  [projectionName, {
    shape: [4, 4],
    dtype: 'F16',
    role: 'matmul',
    layout: 'row',
    size: 32,
    shardIndex: 2,
    offset: 0,
  }],
  [projectionNormName, {
    shape: [4],
    dtype: 'F32',
    role: 'norm',
    layout: 'row',
    size: 16,
    shardIndex: 3,
    offset: 0,
  }],
]);
const shardBytes = new Map([
  [0, Uint8Array.from({ length: 16 }, (_, i) => i)],
  [1, Uint8Array.from({ length: 16 }, (_, i) => i + 16)],
]);
const rangeLoadCalls = [];
const rangeWeights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-range-test',
  perLayerInputSession: {
    materialization: 'auto',
  },
  tensorLocations: rangeTensorLocations,
  async loadTensor(name, toGPU) {
    rangeLoadCalls.push([name, toGPU]);
    if (name === projectionName) {
      return new Float32Array(16);
    }
    if (name === projectionNormName) {
      return new Float32Array(4);
    }
    return null;
  },
  shouldStreamLargeWeight(name) {
    return name === embedName || name === projectionName;
  },
  async loadShardRange(index, offset, length) {
    const bytes = shardBytes.get(index);
    assert.ok(bytes, `missing shard ${index}`);
    return bytes.slice(offset, offset + length).buffer;
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 4,
});

assert.ok(rangeWeights, 'range-backed per-layer input weights should load');
assert.ok(isCpuWeightBuffer(rangeWeights.embedTokensPerLayer), 'range-backed embed tensor should be wrapped as CpuWeightBuffer');
assert.equal(rangeWeights.embedTokensPerLayer.data.kind, 'tensor_range_source');
assert.ok(
  isCpuWeightBuffer(rangeWeights.perLayerModelProjection),
  'range-backed projection tensor should be wrapped as CpuWeightBuffer'
);
assert.equal(rangeWeights.perLayerModelProjection.data.kind, 'tensor_range_source');
assert.deepEqual(rangeLoadCalls, [
  [projectionNormName, true],
]);
const rangeBytes = await rangeWeights.embedTokensPerLayer.data.loadRange(12, 8);
assert.deepEqual(Array.from(rangeBytes), [12, 13, 14, 15, 16, 17, 18, 19]);

const undersizedProjectionLocations = new Map([
  [embedName, {
    shape: [16, 32],
    dtype: 'F16',
    role: 'embedding',
    layout: 'row',
  }],
  [projectionName, {
    shape: [32, 64],
    dtype: 'F16',
    role: 'matmul',
    layout: 'row',
    size: 2048,
    shardIndex: 0,
    offset: 0,
  }],
  [projectionNormName, {
    shape: [32],
    dtype: 'F32',
    role: 'norm',
    layout: 'row',
  }],
]);
const undersizedProjectionWeights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-undersized-projection-test',
  perLayerInputSession: {
    materialization: 'auto',
  },
  tensorLocations: undersizedProjectionLocations,
  async loadTensor(name, toGPU) {
    if (name === embedName) {
      return new Float32Array(16 * 32);
    }
    if (name === projectionName) {
      assert.equal(toGPU, true);
      return {
        buffer: { size: 2048 },
        dtype: 'f16',
        layout: 'row',
        shape: Object.freeze([32, 64]),
        label: projectionName,
        materializations: Object.freeze({
          f16: Object.freeze({
            buffer: { size: 2048 },
            layout: 'row',
          }),
        }),
      };
    }
    if (name === projectionNormName) {
      return new Float32Array(32);
    }
    return null;
  },
  shouldStreamLargeWeight() {
    return false;
  },
  async loadShardRange(_index, offset, length) {
    return Uint8Array.from({ length }, (_, i) => (offset + i) % 251).buffer;
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 32,
});

assert.ok(
  isCpuWeightBuffer(undersizedProjectionWeights.perLayerModelProjection),
  'undersized resident projection should fall back to a range-backed CPU source'
);
assert.equal(
  undersizedProjectionWeights.perLayerModelProjection.data.kind,
  'tensor_range_source'
);

const packedProjectionWeights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-packed-projection-test',
  perLayerInputSession: {
    materialization: 'auto',
  },
  tensorLocations: undersizedProjectionLocations,
  async loadTensor(name, toGPU) {
    if (name === embedName) {
      return new Float32Array(16 * 32);
    }
    if (name === projectionName) {
      if (toGPU === false) {
        return Uint8Array.from({ length: 2048 }, (_, i) => i % 251);
      }
      return {
        buffer: { size: 2048 },
        dtype: 'q4k_m',
        layout: 'row',
        shape: Object.freeze([32, 64]),
        label: projectionName,
        materializations: Object.freeze({
          q4k_m: Object.freeze({
            buffer: { size: 2048 },
            layout: 'row',
          }),
        }),
      };
    }
    if (name === projectionNormName) {
      return new Float32Array(32);
    }
    return null;
  },
  shouldStreamLargeWeight() {
    return false;
  },
  async loadShardRange() {
    throw new Error('packed resident projection should not fall back to range-backed loading');
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 32,
});

assert.ok(
  !isCpuWeightBuffer(packedProjectionWeights.perLayerModelProjection),
  'packed resident projection weights must stay GPU-resident instead of falling back to a range-backed CPU source'
);
assert.equal(
  packedProjectionWeights.perLayerModelProjection.dtype,
  'q4k_m',
  'packed projection weights with non-q4k manifest metadata should preserve the resident materialization'
);

const packedRawProjectionWeights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-packed-raw-projection-test',
  perLayerInputSession: {
    materialization: 'auto',
  },
  tensorLocations: new Map([
    [embedName, {
      shape: [16, 32],
      dtype: 'F16',
      role: 'embedding',
      layout: 'row',
    }],
    [projectionName, {
      shape: [32, 64],
      dtype: 'Q4_K_M',
      role: 'matmul',
      layout: 'row',
      size: 4608,
      shardIndex: 0,
      offset: 0,
    }],
    [projectionNormName, {
      shape: [32],
      dtype: 'F32',
      role: 'norm',
      layout: 'row',
    }],
  ]),
  async loadTensor(name, toGPU) {
    if (name === embedName) {
      return new Float32Array(16 * 32);
    }
    if (name === projectionName) {
      if (toGPU === false) {
        return Uint8Array.from({ length: 4608 }, (_, i) => (i * 7) % 251);
      }
      return {
        __dopplerFakeGPUBuffer: true,
        size: 4608,
        usage: 0,
        destroy() {},
        async mapAsync() {},
        getMappedRange() {
          return new ArrayBuffer(0);
        },
        unmap() {},
      };
    }
    if (name === projectionNormName) {
      return new Float32Array(32);
    }
    return null;
  },
  shouldStreamLargeWeight() {
    return false;
  },
  async loadShardRange() {
    throw new Error('packed raw resident projection should not fall back to range-backed loading');
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 32,
});

assert.ok(
  !isCpuWeightBuffer(packedRawProjectionWeights.perLayerModelProjection),
  'packed raw resident projection buffers must stay resident instead of falling back to a range-backed CPU source'
);
assert.equal(
  packedRawProjectionWeights.perLayerModelProjection.dtype,
  'f32',
  'packed raw q4k projection weights should stabilize to dense f32'
);

const splitTensorLocations = new Map([
  ['model.language_model.layers.0.embed_tokens_per_layer.weight', {
    shape: [16, 32],
    dtype: 'F16',
    role: 'embedding',
    layout: 'row',
  }],
  ['model.language_model.layers.1.embed_tokens_per_layer.weight', {
    shape: [16, 32],
    dtype: 'F16',
    role: 'embedding',
    layout: 'row',
  }],
  [projectionName, {
    shape: [32, 64],
    dtype: 'F16',
    role: 'matmul',
    layout: 'row',
  }],
  [projectionNormName, {
    shape: [32],
    dtype: 'F32',
    role: 'norm',
    layout: 'row',
  }],
]);
const splitLoadCalls = [];
const splitWeights = await loadPerLayerInputWeights({
  modelId: 'gemma4-e2b-split-test',
  perLayerInputSession: {
    materialization: 'gpu_split_tables',
  },
  tensorLocations: splitTensorLocations,
  async loadTensor(name, toGPU) {
    splitLoadCalls.push([name, toGPU]);
    if (name === projectionName) {
      return new Float32Array(32 * 64);
    }
    if (name === projectionNormName) {
      return new Float32Array(32);
    }
    return new Float32Array(16 * 32);
  },
  shouldStreamLargeWeight() {
    return false;
  },
  resolveWeightLayout(loc) {
    return loc.layout ?? 'row';
  },
}, {
  hiddenSizePerLayerInput: 32,
  numLayers: 2,
});

assert.ok(splitWeights, 'split per-layer input weights should load');
assert.equal(splitWeights.embedTokensPerLayerSplit?.length, 2);
assert.ok(isCpuWeightBuffer(splitWeights.embedTokensPerLayer) || splitWeights.embedTokensPerLayer);
assert.deepEqual(
  [...splitLoadCalls].sort(([nameA], [nameB]) => nameA.localeCompare(nameB)),
  [
    ['model.language_model.layers.0.embed_tokens_per_layer.weight', true],
    ['model.language_model.layers.1.embed_tokens_per_layer.weight', true],
    [projectionName, true],
    [projectionNormName, true],
  ].sort(([nameA], [nameB]) => nameA.localeCompare(nameB))
);

console.log('per-layer-input-loader.test: ok');

destroyBufferPool();
setDevice(null, { platformConfig: null });
