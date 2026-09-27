import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { loadLayer } from '../../src/loader/layer-loader.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`layer-loader-affine-norm.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const before = new Float32Array([1, 2, 3, 4]);
const after = new Float32Array([-1, 0, 1, 2]);
const tensors = new Map([
  ['model.layers.0.pre_feedforward_layernorm.bias', before],
  ['model.layers.0.post_feedforward_layernorm.bias', after],
]);
const load = () => loadLayer({
  tensorLocations: new Map(),
  loadTensor: async name => tensors.get(name) ?? null,
  needsNormWeightOffset: () => false,
  gpuBuffers: new Set(),
  keepF32Weights: true,
  isMoE: false,
  isExpertLayer: () => false,
}, 0);
const layer = await load();
assert.equal(layer.preFeedforwardNormBias, before);
assert.equal(layer.postFeedforwardNormBias, after);
tensors.clear();
const missing = await load();
assert.equal(missing.preFeedforwardNormBias, null);
assert.equal(missing.postFeedforwardNormBias, null);
console.log('layer-loader-affine-norm.test: declared affine biases retained; missing biases remain null');
