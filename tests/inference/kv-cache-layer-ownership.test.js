import assert from 'node:assert/strict';
import { createKVCache } from '../../src/inference/pipelines/text/init.js';
import { DEFAULT_KVCACHE_CONFIG } from '../../src/config/schema/index.js';
import { setDevice } from '../../src/gpu/device.js';

const originalGPUBuffer = globalThis.GPUBuffer;
const originalGPUBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08 };
globalThis.GPUBuffer = class GPUBuffer {};
setDevice({ features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
  createBindGroup() { throw new Error('Cache allocation test must not create bind groups.'); },
  createBuffer(descriptor) { return Object.assign(new globalThis.GPUBuffer(), descriptor, { destroy() {} }); },
  queue: { submit() { throw new Error('CPU cache test must not submit GPU work.'); },
    onSubmittedWorkDone: async () => {} } }, { platformConfig: null });

try {
  // LFM2 shape: 16 layers, 10 short-convolution and 6 full attention. Only the
  // six attention layers get a cache slot; the slots map back to model indices.
  const lfm2Types = ['conv', 'conv', 'full_attention', 'conv', 'conv', 'full_attention', 'conv', 'conv', 'full_attention', 'conv', 'full_attention', 'conv', 'full_attention', 'conv', 'full_attention', 'conv'];
  const model = { numLayers: 16, numKVHeads: 1, headDim: 8, maxSeqLen: 16, slidingWindow: null, attnLogitSoftcapping: null, layerTypes: lfm2Types };
  const policy = { ...DEFAULT_KVCACHE_CONFIG, kvDtype: 'f32', maxSeqLen: 16 };
  const cache = createKVCache(model, false, false, policy);
  try {
    assert.equal(cache.numLayers, 6, 'conv layers hold no attention KV');
    assert.deepEqual(cache.layerIndices, [2, 5, 8, 10, 12, 14]);
    assert.throws(() => cache.get(0), /no allocated KV cache/);
  } finally { cache.destroy(); }

  // A model with attention layers only keeps the full allocation.
  const dense = createKVCache({ ...model, numLayers: 4, layerTypes: Array(4).fill('full_attention') }, false, false, policy);
  try { assert.equal(dense.numLayers, 4); } finally { dense.destroy(); }

  // Mixed types must be declared for every layer.
  assert.throws(() => createKVCache({ ...model, layerTypes: ['conv', 'full_attention'] }, false, false, policy), /explicit types for every model layer/);
  console.log('kv cache layer ownership ok');
} finally {
  globalThis.GPUBuffer = originalGPUBuffer;
  globalThis.GPUBufferUsage = originalGPUBufferUsage;
}
