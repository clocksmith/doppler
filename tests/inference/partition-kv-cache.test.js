import assert from 'node:assert/strict';
import { createKVCache } from '../../src/inference/pipelines/text/init.js';
import { DEFAULT_KVCACHE_CONFIG } from '../../src/config/schema/index.js';
import { SlidingWindowKVCache } from '../../src/inference/kv-cache/sliding-window.js';
import { setDevice } from '../../src/gpu/device.js';
const originalGPUBuffer = globalThis.GPUBuffer;
globalThis.GPUBuffer = class GPUBuffer {};

setDevice({ features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
  createBindGroup() { throw new Error('CPU cache test must not create bind groups.'); },
  queue: { submit() { throw new Error('CPU cache test must not submit GPU work.'); },
    onSubmittedWorkDone: async () => {} } }, { platformConfig: null });

const model = { numLayers: 4, numKVHeads: 1, headDim: 8, maxSeqLen: 16,
  slidingWindow: 4, attnLogitSoftcapping: null,
  layerTypes: ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'] };
const policy = { ...DEFAULT_KVCACHE_CONFIG, kvDtype: 'f32', maxSeqLen: 16 };
const full = createKVCache(model, false, false, policy);
const a = createKVCache(model, false, false, policy, [0, 1]);
const b = createKVCache(model, false, false, policy, [2, 3]);
try {
  assert.equal(a.numLayers, 2);
  assert.equal(b.numLayers, 2);
  assert.equal(a.layout, full.layout);
  assert.equal(a.maxSeqLen, full.maxSeqLen, 'A must retain full-model cache policy despite owning only local-attention layers');
  assert.equal(a.getMemoryStats().allocated + b.getMemoryStats().allocated, full.getMemoryStats().allocated);
  const keys = Float32Array.from({ length: 8 }, (_, i) => i + 1);
  const values = Float32Array.from(keys, v => v * 2);
  for (const layer of [2, 3]) b.update(layer, keys, values, 0);
  assert.equal(b.currentSeqLen, 1, 'Original last-layer index advances local cache position');
  assert.deepEqual(b.get(3, 0, 1).keys, keys);
  assert.equal(a.currentSeqLen, 0);
  assert.throws(() => b.get(1), /outside resident KV range/);
  assert.throws(() => b.getGPUBuffers(4), /outside resident KV range/);
  assert.throws(() => b.update(0, keys, values, 0), /outside resident KV range/);
  assert.throws(() => { b.currentSeqLen = 4; }, /owned by the cache/);
  b.clear();
  assert.equal(b.currentSeqLen, 0);
  const sliding = createKVCache({ ...model, layerTypes: model.layerTypes.map(() => 'sliding_attention') },
    false, false, policy, [2, 3]);
  try {
    assert.ok(sliding instanceof SlidingWindowKVCache);
    assert.throws(() => sliding.getGPUBuffers(1), /outside resident KV range/);
    assert.equal(sliding.getGPUBuffers(2), null);
  } finally { sliding.destroy(); }
  for (const range of [[-1, 0], [0, 4], [3, 2], [0.5, 1], [0]]) {
    assert.throws(() => createKVCache(model, false, false, policy, range), /inclusive range/);
  }
  assert.throws(() => createKVCache({ ...model, globalHeadDim: 16 }, false, false, policy, [2, 3]), /mixed geometry/);
} finally {
  full.destroy(); a.destroy(); b.destroy(); setDevice(null);
  if (originalGPUBuffer === undefined) delete globalThis.GPUBuffer;
  else globalThis.GPUBuffer = originalGPUBuffer;
}
console.log('partition-kv-cache: range allocation, model policy, isolation, bounds, subclass identity, and cleanup passed');
