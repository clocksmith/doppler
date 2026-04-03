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
  [embedName, false],
  [projectionName, true],
  [projectionNormName, true],
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
    return name === embedName;
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
assert.deepEqual(rangeLoadCalls, [
  [projectionName, true],
  [projectionNormName, true],
]);
const rangeBytes = await rangeWeights.embedTokensPerLayer.data.loadRange(12, 8);
assert.deepEqual(Array.from(rangeBytes), [12, 13, 14, 15, 16, 17, 18, 19]);

console.log('per-layer-input-loader.test: ok');
