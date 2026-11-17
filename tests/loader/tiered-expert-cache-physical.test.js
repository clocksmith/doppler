import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { getDeviceMemorySnapshot, setDeviceMemoryBudget } from '../../src/memory/device-budget.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { TieredExpertCache } from '../../src/loader/experts/tiered-expert-cache.js';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';

// Synthetic source bytes exercise physical copy/lifetime only. Authentication
// and numerical model qualification are separate complete-request gates.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, `Physical tiered check unavailable: ${probe.reason}`);
assert.ok(!/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(getKernelCapabilities()?.adapterInfo)));
const device = getDevice(), capBytes = 8 * 1024 * 1024;
setDeviceMemoryBudget(device, capBytes);
const errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const source = Uint8Array.from({ length: 288 }, (_, i) => i % 251);
const plans = Array.from({ length: 3 }, (_, layer) => ({ layer, numLayers: 3, numExperts: 4,
  hiddenSize: 1, intermediateSize: 2, expertBytes: 24, bankBytes: 96,
  planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: `${layer}_${role}`,
    expertBytes: 8, bytes: 32, location: { shardIndex: 0, offset: layer * 96 + i * 32, size: 32, shape: [4, 2, 1] } })) }));
const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
  storageMode: 'tiered-compact', tiered: { residentLayers: [0], parkedLayers: [1], mappedCacheBytes: 48, mappedProtectedShare: 0 },
  sourceReadMode: 'expert-lease-batch', cacheBytes: 368, compactLayerBytes: { 1: 64, 2: 64 },
  stagingBytes: 256, readConcurrency: 2, gpuBudgetBytes: capBytes, reserveBytes: 1024 };
let reads = 0, leases = 0, beforeClose;
const cache = new TieredExpertCache({ device, policy, otherGpuBytes: () => getBufferPool(device).getCurrentBytesAllocated(),
  readRange: async () => { throw Error('Unowned source fallback'); },
  rangeLease: { maxPartBytes: 24, memoryBytes: 80, concurrency: 2,
    read: async () => { throw Error('Whole expert batch required'); },
    async readMany(parts) {
      reads++;
      const bytes = new Uint8Array(24);
      let offset = 0;
      for (const part of parts) { bytes.set(source.subarray(part.offset, part.offset + part.length), offset); offset += part.length; }
      leases++;
      return { bytes, release() { leases--; structuredClone(bytes.buffer, { transfer: [bytes.buffer] }); } };
    } } }, plans, 2);
async function verify(weights, layer, ids) {
  const map = weights.sourceExpertToSlot ? new Uint32Array(await readBuffer(weights.sourceExpertToSlot, 16)) : null;
  for (const [i, role] of ['gate', 'up', 'down'].entries()) {
    const bytes = new Uint8Array(await readBuffer(weights[role].buffer, weights[role].buffer.size));
    for (const expert of ids) {
      const slot = map ? map[expert] : expert;
      assert.deepEqual(bytes.slice(slot * 8, slot * 8 + 8), source.slice(layer * 96 + i * 32 + expert * 8, layer * 96 + i * 32 + expert * 8 + 8));
    }
  }
}
try {
  await cache.preload();
  assert.equal(reads, 8);
  await verify(cache.getResidentWeights(0), 0, [0, 3]);
  await verify(await cache.prepare(plans[1], [3, 1], undefined, 'prefill'), 1, [3, 1]);
  assert.equal(reads, 8, 'parked GPU copies do not acquire source bytes again');
  await verify(await cache.prepare(plans[2], [0, 1], undefined, 'prefill'), 2, [0, 1]);
  await verify(await cache.prepare(plans[2], [2, 3], undefined, 'decode'), 2, [2, 3]);
  const abort = new AbortController(); abort.abort();
  await assert.rejects(cache.prepare(plans[2], [0], abort.signal, 'decode'), /abort/i);
  await verify(await cache.prepare(plans[2], [0, 1], undefined, 'decode'), 2, [0, 1]);
  beforeClose = cache.getStats();
  assert.ok(beforeClose.tiered.mapped.reuses > 0, 'streamed refill reuses idle mapped allocations');
  assert.ok(beforeClose.peakResidentBytes <= policy.cacheBytes);
  assert.ok(beforeClose.peakStagingBytes <= policy.stagingBytes);
  assert.equal(leases, 0);
} finally {
  try {
    await cache.destroy();
    const pool = getBufferPool(device); pool.destroy(); await pool.settleDeferredDestruction();
    const retired = getDeviceMemorySnapshot(device);
    assert.equal(retired.liveBytes, 0);
    assert.ok(retired.peakBytes <= capBytes);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ test: 'tiered-expert-cache-physical', beforeClose, reads, leases,
      apiCapBytes: capBytes, peakApiBytes: retired.peakBytes, retiredLiveBytes: retired.liveBytes,
      scope: 'synthetic resident/parked/streamed copies, cancel/reuse and retirement; no model/authentication claim' }));
  } finally { destroyDevice(); await releaseNodeWebGPU(); }
}
