import assert from 'node:assert/strict';
import test from 'node:test';
import { TieredExpertCache } from '../../src/loader/experts/tiered-expert-cache.js';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';
import { markDeviceLost } from '../../src/memory/device-residency.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_WRITE: 2 };
const turn = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
    storageMode: 'tiered-compact', tiered: { residentLayers: [0], parkedLayers: [1], mappedCacheBytes: 96 },
    sourceReadMode: 'expert-lease-batch', cacheBytes: 416, compactLayerBytes: { 1: 64, 2: 64 },
    stagingBytes: 256, readConcurrency: 2, gpuBudgetBytes: 1024, reserveBytes: 128 };
  const source = Uint8Array.from({ length: 288 }, (_, i) => i % 251);
  const plans = Array.from({ length: 3 }, (_, layer) => ({ layer, numLayers: 3, numExperts: 4,
    hiddenSize: 1, intermediateSize: 2, expertBytes: 24, bankBytes: 96,
    planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: `${layer}_${role}`,
      expertBytes: 8, bytes: 32, location: { shardIndex: 0, offset: layer * 96 + i * 32, size: 32, shape: [4, 2, 1] } })) }));
  const state = { live: 0, reads: 0, leases: 0, failRead: false, failDispose: false, waitRead: null,
    submitted: 0, writes: 0, activeReads: 0, peakReads: 0, failFence: false, fence: Promise.resolve() };
  const device = { limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024 },
    lost: new Promise(() => {}),
    createBuffer(descriptor) {
      state.live += descriptor.size;
      return { ...descriptor, data: new Uint8Array(descriptor.size), destroyed: false,
        mapped: descriptor.mappedAtCreation === true,
        getMappedRange() { assert(this.mapped); return this.data.buffer; },
        unmap() { this.data = new Uint8Array(structuredClone(this.data.buffer, { transfer: [this.data.buffer] })); this.mapped = false; },
        destroy() {
          if (state.failDispose) throw Error('dispose failed');
          if (!this.destroyed) state.live -= this.size;
          this.destroyed = true;
        } };
    },
    createCommandEncoder() {
      const ops = [];
      return { copyBufferToBuffer(source, from, target, to, bytes) {
        ops.push(() => { assert(!source.destroyed && !source.mapped); target.data.set(source.data.subarray(from, from + bytes), to); });
      }, finish: () => ops };
    },
    queue: { writeBuffer(target, offset, data) {
      state.writes++;
      target.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    }, submit(commands) { state.submitted++; for (const ops of commands) for (const op of ops) op(); },
    onSubmittedWorkDone() { return state.failFence ? Promise.reject(Error('fence failed')) : state.fence; } } };
  const options = { device, policy, otherGpuBytes: () => 0,
    readRange: async () => { throw Error('Unowned fallback must not run'); },
    rangeLease: { maxPartBytes: 24, memoryBytes: 80, concurrency: 2,
      read: async () => { throw Error('Whole expert batches required'); },
      async readMany(parts) {
        state.reads++; state.activeReads++; state.peakReads = Math.max(state.peakReads, state.activeReads);
        try {
          await state.waitRead;
          if (state.failRead) throw Error('source authentication failed');
          const bytes = new Uint8Array(parts.reduce((sum, row) => sum + row.length, 0));
          let offset = 0;
          for (const part of parts) { bytes.set(source.subarray(part.offset, part.offset + part.length), offset); offset += part.length; }
          state.leases++;
          return { bytes, release() { state.leases--; structuredClone(bytes.buffer, { transfer: [bytes.buffer] }); } };
        } finally { state.activeReads--; }
      } } };
  const cache = new TieredExpertCache(options, plans, 2);
  const verify = (weights, layer, expert) => {
    const slot = weights.sourceExpertToSlot ? new Uint32Array(weights.sourceExpertToSlot.data.buffer)[expert] : expert;
    for (const [i, role] of ['gate', 'up', 'down'].entries()) assert.deepEqual(
      weights[role].buffer.data.slice(slot * 8, slot * 8 + 8), source.slice(layer * 96 + i * 32 + expert * 8, layer * 96 + i * 32 + expert * 8 + 8));
  };
  return { cache, state, plans, options, verify };
}

test('resident setup and parked/T1 stream copies preserve source expert identity in both phases', async () => {
  const { cache, state, plans, verify } = fixture();
  assert.equal(cache.getResidentWeights(0), null);
  await assert.rejects(cache.prepare(plans[2], [0, 1], undefined, 'prefill'), /completed setup/);
  await cache.preload();
  assert.equal(state.reads, 8);
  assert.equal(state.leases, 0);
  assert.equal(state.peakReads, 2);
  for (const expert of [0, 3]) verify(cache.getResidentWeights(0), 0, expert);
  assert.equal(cache.getResidentWeights(0).sourceExpertToSlot, undefined, 'resident kernels use original IDs without a CPU cut');
  const parked = await cache.prepare(plans[1], [3, 1], undefined, 'prefill');
  verify(parked, 1, 3); verify(parked, 1, 1);
  assert.equal(state.reads, 8, 'parked full banks are read once in setup');
  const streamed = await cache.prepare(plans[2], [0, 1], undefined, 'prefill');
  verify(streamed, 2, 0); verify(streamed, 2, 1);
  await cache.prepare(plans[2], [2, 3], undefined, 'decode');
  const fresh = await cache.prepare(plans[2], [0, 1], undefined, 'decode');
  verify(fresh, 2, 0); verify(fresh, 2, 1);
  assert.equal(state.reads, 12, 'T1 hits refill evicted GPU slots without another authenticated source acquisition');
  assert.equal(state.leases, 0);
  assert.equal(cache.getStats().tiered.mapped.pinnedBytes, 96);
  assert(cache.getStats().residentBytes <= 416);
  assert(cache.getStats().peakStagingBytes <= 256);
  assert.equal(cache.capacity(2), 2);
  assert.equal(cache.capacity(0), 4);
  assert.equal(cache.capacity(99), null);
  await assert.rejects(cache.prepare({ ...plans[2] }, [0], undefined, 'decode'), /admitted source plan/);
  await cache.destroy();
  assert.equal(state.live, 0);
  assert.equal(cache.getStats().residentBytes, 0);
});

test('cancelled and rejected streamed reads settle leases before reuse and retryable unload', async () => {
  const { cache, state, plans } = fixture();
  await cache.preload();
  state.failRead = true;
  await assert.rejects(cache.prepare(plans[2], [0, 1], undefined, 'decode'), /authentication/);
  assert.equal(state.activeReads, 0);
  assert.equal(state.leases, 0);
  state.failRead = false;
  let release;
  state.waitRead = new Promise(resolve => { release = resolve; });
  const abort = new AbortController();
  const active = cache.prepare(plans[2], [0, 1], abort.signal, 'prefill');
  await turn(); abort.abort(); release();
  await assert.rejects(active, /abort/i);
  assert.equal(state.leases, 0);
  state.waitRead = null;
  await cache.prepare(plans[2], [0, 1], undefined, 'decode');
  state.failDispose = true;
  await assert.rejects(cache.destroy(), /cleanup failed/);
  assert(state.live > 0);
  state.failDispose = false;
  await cache.destroy();
  assert.equal(state.live, 0);
});

test('close cancels setup and waits cancellation-ignoring source reads before releasing allocations', async () => {
  const { cache, state } = fixture();
  let release;
  state.waitRead = new Promise(resolve => { release = resolve; });
  const loading = cache.preload();
  await turn();
  let closed = false;
  const closing = cache.destroy().then(() => { closed = true; });
  await turn(); assert.equal(closed, false);
  release();
  await assert.rejects(loading, /unloaded/);
  await closing;
  assert.equal(state.live, 0);
  assert.equal(state.leases, 0);
  assert.equal(cache.getResidentWeights(0), null);
});

test('all-resident placement rejects invalid lease lanes before allocating or publishing empty banks', async () => {
  const { cache, state, plans, options } = fixture();
  const policy = { ...options.policy, compactLayerBytes: null,
    tiered: { residentLayers: [0, 1, 2], parkedLayers: [], mappedCacheBytes: 0 } };
  for (const change of [{ concurrency: 0 }, { concurrency: 1.5 }, { maxPartBytes: Infinity },
    { maxPartBytes: 25 }, { memoryBytes: 4 }]) {
    assert.throws(() => new TieredExpertCache({ ...options, policy,
      rangeLease: { ...options.rangeLease, ...change } }, plans, 2), /stagingBytes/);
    assert.equal(state.live, 0);
    assert.equal(state.reads, 0);
  }
  await cache.destroy();
});

test('ordinary failed fences retain ownership, while confirmed device loss permits complete retirement', async () => {
  const { cache, state, plans, options } = fixture();
  await cache.preload();
  await cache.prepare(plans[2], [0, 1], undefined, 'decode');
  state.failFence = true;
  await assert.rejects(cache.destroy(), /cleanup failed/);
  assert(state.live > 0, 'an ordinary rejected fence cannot release still-owned resident banks');
  markDeviceLost(options.device);
  await cache.destroy();
  assert.equal(state.live, 0);
  assert.equal(cache.getStats().residentBytes, 0);
  assert.equal(cache.getResidentWeights(0), null);
});

test('resident setup bounds queued upload payloads by retaining every lane until its actual fence', async () => {
  const { cache, state } = fixture();
  let complete;
  state.fence = new Promise(resolve => { complete = resolve; });
  const loading = cache.preload();
  await turn();
  assert.equal(state.reads, 2, 'no lane reads another expert while its native upload is outstanding');
  assert.equal(state.leases, 2, 'leases stay charged through the bounded upload drain');
  assert.equal(cache.getStats().peakStagingBytes, 128, '80-byte reader reservation plus two24-byte upload payloads');
  assert.equal(cache.getResidentWeights(0), null);
  complete(); await loading;
  assert.equal(state.leases, 0);
  await cache.destroy();
  assert.equal(state.live, 0);
});
