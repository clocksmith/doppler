import assert from 'node:assert/strict';

globalThis.GPUBufferUsage ??= {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
};

class FakeBuffer {
  constructor({ size, usage, label = '' }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
  }

  destroy() {}
}

const { setDevice } = await import('../../src/gpu/device.js');
import { loadLayer } from '../../src/loader/layer-loader.js';

setDevice({
  features: new Set(),
  limits: {
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384,
  },
  queue: {
    submit() {},
    writeBuffer() {},
  },
  createBuffer(options) {
    return new FakeBuffer(options);
  },
  createBindGroup() {
    return {};
  },
  createCommandEncoder() {
    return { finish: () => ({}) };
  },
  createShaderModule() {
    return {};
  },
  lost: new Promise(() => {}),
}, { platformConfig: null });

const inputNorm = Float32Array.of(1);
const postSelfAttentionNorm = Float32Array.of(2);
const preFeedforwardNorm = Float32Array.of(3);
const postFeedforwardNorm = Float32Array.of(4);
const tensors = new Map([
  ['model.language_model.layers.0.input_layernorm.weight', inputNorm],
  ['model.language_model.layers.0.post_self_attn_layernorm.weight', postSelfAttentionNorm],
  ['model.language_model.layers.0.post_attention_layernorm.weight', preFeedforwardNorm],
  ['model.language_model.layers.0.post_mlp_layernorm.weight', postFeedforwardNorm],
]);

const weights = await loadLayer({
  tensorLocations: new Map(),
  loadTensor: async (name) => tensors.get(name) ?? null,
  needsNormWeightOffset: () => false,
  gpuBuffers: new Set(),
  keepF32Weights: true,
  isMoE: false,
  isExpertLayer: () => false,
}, 0);

assert.equal(weights.inputNorm, inputNorm);
assert.equal(weights.postAttentionNorm, postSelfAttentionNorm);
assert.equal(weights.preFeedforwardNorm, preFeedforwardNorm);
assert.equal(weights.postFeedforwardNorm, postFeedforwardNorm);
assert.equal(weights.postAttnNorm, postSelfAttentionNorm);

setDevice(null, { platformConfig: null });

console.log('glmocr-layer-norm-aliases.test: ok');
