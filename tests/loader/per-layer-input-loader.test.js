import assert from 'node:assert/strict';

import { loadPerLayerInputWeights } from '../../src/loader/per-layer-input-loader.js';
import { isCpuWeightBuffer } from '../../src/gpu/weight-buffer.js';

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
assert.deepEqual(loadCalls, [
  [projectionName, true],
  [projectionNormName, true],
  [embedName, false],
]);
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
assert.deepEqual(splitLoadCalls, [
  ['model.language_model.layers.0.embed_tokens_per_layer.weight', true],
  ['model.language_model.layers.1.embed_tokens_per_layer.weight', true],
  [projectionName, true],
  [projectionNormName, true],
]);

console.log('per-layer-input-loader.test: ok');
