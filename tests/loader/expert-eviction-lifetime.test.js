import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 0x40, STORAGE: 0x80 };

const { ExpertCache } = await import('../../src/loader/experts/expert-cache.js');
const { isExpertResident } = await import('../../src/loader/experts/expert-loader.js');
const { ensureExpertLoaded } = await import('../../src/inference/pipelines/text/moe/expert-loading.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { acquireBuffer, destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { setDevice } = await import('../../src/gpu/device.js');

class FakeBuffer {
  constructor({ size, usage }) { this.size = size; this.usage = usage; this.destroyed = false; }
  destroy() { this.destroyed = true; }
}

setDevice({
  features: new Set(),
  limits: { maxStorageBufferBindingSize: 1 << 20, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
  queue: { submit() {}, onSubmittedWorkDone: () => Promise.resolve() },
  createBindGroup: descriptor => descriptor,
  createBuffer: descriptor => new FakeBuffer(descriptor),
}, { platformConfig: null });
getBufferPool().configure({ enablePooling: false });

// Mixtral experts are LRU cached. Two 20-byte experts exceed a 32-byte cache's
// capacity, so loading expert 1 evicts and releases expert 0.
const ctx = {
  manifest: { modelId: 'eviction-fixture', moeConfig: { expertFormat: 'mixtral' } },
  expertCache: new ExpertCache(32, { evictionHighWatermark: 0.9, emergencyTrimToRatio: 0.75 }),
  experts: new Map(),
};
let loads = 0;
const loader = {
  isExpertResident: (layer, expert, weights) => isExpertResident(ctx, layer, expert, weights),
  async loadExpert(layer, expert) {
    const cached = ctx.expertCache.get(layer, expert);
    if (cached) return cached;
    loads += 1;
    const buffer = acquireBuffer(16, GPUBufferUsage.STORAGE, `expert_${expert}`);
    const weights = { gate: createWeightBuffer(buffer, 'f16', 'row', [1, 8], `expert_${expert}`), up: null, down: null,
      expertFormat: 'mixtral' };
    ctx.expertCache.put(layer, expert, weights, 20);
    return weights;
  },
};

const consumer = new Map();
await ensureExpertLoaded(0, 0, consumer, loader);
const first = consumer.get('layer_0_expert_0');
await ensureExpertLoaded(0, 1, consumer, loader);
await Promise.resolve();
await Promise.resolve();
assert.equal(ctx.expertCache.has(0, 0), false, 'fixture: loading expert 1 evicts expert 0');
assert.equal(first.gate.buffer.destroyed, true, 'fixture: loading expert 1 evicts and releases expert 0');

await ensureExpertLoaded(0, 0, consumer, loader);
const revisited = consumer.get('layer_0_expert_0');
assert.notEqual(revisited, first, 'an evicted expert is reloaded, not served from the consumer map');
assert.equal(revisited.gate.buffer.destroyed, false, 'dispatch never sees released expert buffers');
assert.equal(loads, 3);

// Resident experts are not reloaded on every use.
await ensureExpertLoaded(0, 0, consumer, loader);
assert.equal(consumer.get('layer_0_expert_0'), revisited);
assert.equal(loads, 3);

// Packed formats are never LRU evicted; the loader keeps them until unload.
assert.equal(isExpertResident({ ...ctx, manifest: { moeConfig: { expertFormat: 'gpt-oss' } } }, 0, 0, {}), true);

destroyBufferPool();
setDevice(null);
console.log('expert-eviction-lifetime.test: ok');
