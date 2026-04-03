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

console.log('per-layer-input-loader.test: ok');
