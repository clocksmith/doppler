import assert from 'node:assert/strict';
import { createKVCache } from '../../src/inference/pipelines/text/init.js';
import { DEFAULT_KVCACHE_CONFIG } from '../../src/config/schema/index.js';
import { SlidingWindowKVCache } from '../../src/inference/kv-cache/sliding-window.js';
import { setDevice } from '../../src/gpu/device.js';
import { captureKvCacheByteProof } from '../../src/tooling/browser-harness/text-evidence.js';
const originalGPUBuffer = globalThis.GPUBuffer;
const originalGPUBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08 };
globalThis.GPUBuffer = class GPUBuffer {};

setDevice({ features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
  createBindGroup() { throw new Error('Cache allocation test must not create bind groups.'); },
  createBuffer(descriptor) { return Object.assign(new globalThis.GPUBuffer(), descriptor, { destroy() {} }); },
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
  const hybridModel = { ...model, slidingWindow: null,
    layerTypes: ['linear_attention', 'full_attention', 'linear_attention', 'full_attention'] };
  const hybrid = createKVCache(hybridModel, false, false, policy);
  const hybridPart = createKVCache(hybridModel, false, false, policy, [2, 3]);
  try {
    assert.equal(hybrid.numLayers, 2, 'Linear recurrence layers own no attention KV allocation');
    assert.equal(hybridPart.numLayers, 1);
    assert.deepEqual(hybrid.layerIndices, [1, 3]);
    assert.deepEqual(hybridPart.layerIndices, [3]);
    assert.throws(() => hybrid.layerIndices.push(9), TypeError);
    assert.equal(hybrid.getMemoryStats().allocated, full.getMemoryStats().allocated / 2);
    for (const layer of [1, 3]) hybrid.update(layer, keys, values, 0);
    hybridPart.update(3, keys, values, 0);
    assert.equal(hybrid.currentSeqLen, 1);
    assert.equal(hybridPart.currentSeqLen, 1);
    assert.deepEqual(hybrid.get(3, 0, 1).keys, keys);
    assert.deepEqual(hybridPart.get(3, 0, 1).values, values);
    for (const layer of [0, 2, 4]) assert.throws(() => hybrid.get(layer), /no allocated KV cache/);
    assert.throws(() => hybridPart.update(1, keys, values, 0), /no allocated KV cache/);
    const proof = await captureKvCacheByteProof({ kvCache: hybrid }, true);
    assert.deepEqual(proof.layers.map(layer => layer.layer), [1, 3]);
    const cloned = hybrid.clone();
    try {
      assert.deepEqual(cloned.get(3, 0, 1).keys, keys, 'Cloning preserves original model indices');
      assert.throws(() => cloned.get(2), /no allocated KV cache/);
    } finally { cloned.destroy(); }
    hybrid.clear();
    assert.equal(hybrid.currentSeqLen, 0);
    assert.equal(hybridPart.currentSeqLen, 1, 'Clearing one cache does not clear another owner');
  } finally { hybrid.destroy(); hybridPart.destroy(); }
  const hybridGPU = createKVCache(hybridModel, true, false, policy);
  const linearOnly = createKVCache({ ...hybridModel, layerTypes: Array(4).fill('linear_attention') },
    true, false, policy, [0, 1]);
  try {
    assert.equal(hybridGPU.numLayers, 2);
    assert.equal(hybridGPU.getMemoryStats().allocated, full.getMemoryStats().allocated / 2);
    assert.ok(hybridGPU.getGPUBuffers(1).keysGPU instanceof globalThis.GPUBuffer);
    assert.ok(hybridGPU.getGPUBuffers(3).valuesGPU instanceof globalThis.GPUBuffer);
    assert.throws(() => hybridGPU.getGPUBuffers(0), /no allocated KV cache/);
    assert.equal(linearOnly.numLayers, 0);
    assert.equal(linearOnly.getMemoryStats().allocated, 0);
    assert.throws(() => linearOnly.getGPUBuffers(0), /no allocated KV cache/);
    linearOnly.clear();
    assert.equal(linearOnly.currentSeqLen, 0);
  } finally { hybridGPU.destroy(); linearOnly.destroy(); }
  assert.throws(() => createKVCache({ ...hybridModel, layerTypes: ['linear_attention'] },
    false, false, policy), /explicit types for every/);
  for (const range of [[-1, 0], [0, 4], [3, 2], [0.5, 1], [0]]) {
    assert.throws(() => createKVCache(model, false, false, policy, range), /inclusive range/);
  }
  assert.throws(() => createKVCache({ ...model, globalHeadDim: 16 }, false, false, policy, [2, 3]), /mixed geometry/);
} finally {
  full.destroy(); a.destroy(); b.destroy(); setDevice(null);
  if (originalGPUBuffer === undefined) delete globalThis.GPUBuffer;
  else globalThis.GPUBuffer = originalGPUBuffer;
  if (originalGPUBufferUsage === undefined) delete globalThis.GPUBufferUsage;
  else globalThis.GPUBufferUsage = originalGPUBufferUsage;
}
console.log('partition-kv-cache: range allocation, model policy, isolation, bounds, subclass identity, and cleanup passed');
