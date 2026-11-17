import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { CompactExpertCache } from '../../src/loader/experts/compact-expert-cache.js';
import { createRoutedExpertCounters } from '../../src/loader/experts/routed-expert-source.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { DEFAULT_EXPERT_CACHE_CONFIG, validateQwen35RoutedConfig } from '../../src/config/schema/runtime/loading.schema.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
test('bounded multi-expert acquisition keeps exact expert/plane order, upload lifetime and failure recovery', { timeout: 5000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'multi-expert-'));
  const source = Uint8Array.from({ length: 192 }, (_, i) => i + 1);
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: source.length }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: hash(manifestBytes), maxPieceBytes: 16,
    files: [{ path: 'weights.bin', size: source.length, pieces: Array.from({ length: 24 }, (_, i) => ({
      offset: i * 8, size: 8, identity: hash(source.subarray(i * 8, i * 8 + 8)),
    })) }] });
  const plan = { layer: 0, numLayers: 1, numExperts: 8, expertBytes: 24, bankBytes: 192,
    intermediateSize: 2, hiddenSize: 1, planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: role,
      expertBytes: 8, bytes: 64, location: { shardIndex: 0, offset: i * 64, size: 64, shape: [8, 2, 1] } })) };
  const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
    storageMode: 'compact-decode', sourceReadMode: 'expert-lease-batch', expertBatchSize: 3,
    cacheBytes: 256, stagingBytes: 320, readConcurrency: 2, gpuBudgetBytes: 4096, reserveBytes: 16 };
  const state = { calls: [], leases: 0, writes: 0, failAt: null, abortAfterRead: null, buffers: [] };
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 }, createBuffer(description) {
    const buffer = { ...description, data: new Uint8Array(description.size), destroyed: false, destroy() { this.destroyed = true; } };
    state.buffers.push(buffer); return buffer;
  }, queue: { async onSubmittedWorkDone() {}, writeBuffer(target, offset, bytes) {
    assert(bytes.byteLength > 0, 'every sibling view remains owned through upload');
    if (++state.writes === state.failAt) throw Error('injected grouped upload failure');
    target.data.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), offset);
  } } };
  let worker, cache;
  try {
    await writeFile(join(root, 'weights.bin'), source);
    worker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: hash(indexBytes),
      files: { 'weights.bin': join(root, 'weights.bin') }, maxPartBytes: 72, maxRanges: 9, concurrency: 2, memoryBudgetBytes: 208 });
    const rangeLease = { ...worker.rangeLease, async readMany(ranges, signal) {
      state.calls.push(ranges.length);
      const lease = await worker.rangeLease.readMany(ranges, signal); state.leases++;
      state.abortAfterRead?.();
      return { bytes: lease.bytes, release() { state.leases--; lease.release(); } };
    } };
    const options = { policy, device, compactLayerCount: 1, otherGpuBytes: () => 0, rangeLease,
      readRange: async () => { throw Error('unowned fallback must not run'); } };
    const stats = createRoutedExpertCounters(); cache = new CompactExpertCache(options, stats);
    const verify = (weights, ids) => {
      const slots = new Uint32Array(weights.sourceExpertToSlot.data.buffer);
      for (const expert of ids) for (const [i, role] of ['gate', 'up', 'down'].entries()) {
        assert.deepEqual(weights[role].buffer.data.slice(slots[expert] * 8, slots[expert] * 8 + 8),
          source.slice(i * 64 + expert * 8, i * 64 + expert * 8 + 8));
      }
    };
    const ids = [7, 2, 5, 0];
    verify(await cache.prepare(plan, ids, new AbortController().signal), ids);
    assert.deepEqual(state.calls, [6, 6], 'four misses balance two experts per admitted lane');
    assert.equal(stats.sourceReads, 12); assert.equal(stats.sourceReadBytes, 96);
    assert.equal(state.leases, 0); assert.equal(stats.stagingAllocations, 0);
    await cache.destroy();
    state.calls.length = 0;
    let releaseReads;
    const bothStarted = new Promise(resolve => { releaseReads = resolve; });
    const balancedLease = { ...rangeLease, async readMany(ranges, signal) {
      state.calls.push(ranges.length);
      if (state.calls.length === 2) releaseReads();
      await bothStarted;
      return worker.rangeLease.readMany(ranges, signal);
    } };
    cache = new CompactExpertCache({ ...options, rangeLease: balancedLease }, createRoutedExpertCounters());
    verify(await cache.prepare(plan, [7, 2], new AbortController().signal), [7, 2]);
    assert.deepEqual(state.calls, [3, 3], 'two misses start on separate lanes before either finishes');
    const callsBeforeHit = state.calls.length;
    verify(await cache.prepare(plan, [2, 7], new AbortController().signal), [2, 7]);
    assert.equal(state.calls.length, callsBeforeHit, 'a fully resident selection issues no zero-sized group');
    await cache.destroy();
    state.calls.length = 0;
    const planeWorker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: hash(indexBytes),
      files: { 'weights.bin': join(root, 'weights.bin') }, maxPartBytes: 72, maxRanges: 9, concurrency: 3, memoryBudgetBytes: 312 });
    let planeCache, releasePlanes, abortPlane;
    const planesStarted = new Promise(resolve => { releasePlanes = resolve; });
    const partReads = [], partLeases = [];
    const planeLease = { ...planeWorker.rangeLease, async readMany(ranges, signal) {
      partReads.push(ranges.length);
      if (partReads.length === 3) releasePlanes();
      await planesStarted;
      const lease = await planeWorker.rangeLease.readMany(ranges, signal);
      partLeases.push(lease); abortPlane?.(); return lease;
    } };
    const planeOptions = { ...options, policy: { ...policy, readConcurrency: 3, stagingBytes: 528 }, rangeLease: planeLease };
    try {
      planeCache = new CompactExpertCache(planeOptions, createRoutedExpertCounters());
      verify(await planeCache.prepare(plan, [7], new AbortController().signal), [7]);
      assert.deepEqual(partReads, [1, 1, 1], 'one miss starts all authenticated planes before any completes');
      assert(partLeases.every(lease => lease.bytes.byteLength === 0));
      await planeCache.destroy(); planeCache = new CompactExpertCache(planeOptions, createRoutedExpertCounters());
      const corrupted = source.slice(); corrupted[184] ^= 128;
      await writeFile(join(root, 'weights.bin'), corrupted);
      const beforeCorrupt = state.writes;
      await assert.rejects(planeCache.prepare(plan, [7], new AbortController().signal), /integrity/);
      assert.equal(state.writes, beforeCorrupt, 'every plane authenticates before ANY plane upload or map publication');
      assert(partLeases.every(lease => lease.bytes.byteLength === 0));
      await writeFile(join(root, 'weights.bin'), source);
      verify(await planeCache.prepare(plan, [7], new AbortController().signal), [7]);
      await planeCache.destroy(); planeCache = new CompactExpertCache(planeOptions, createRoutedExpertCounters());
      state.failAt = state.writes + 2;
      await assert.rejects(planeCache.prepare(plan, [7], new AbortController().signal), /grouped upload/);
      state.failAt = null; assert(partLeases.every(lease => lease.bytes.byteLength === 0));
      verify(await planeCache.prepare(plan, [7], new AbortController().signal), [7]);
      await planeCache.destroy(); planeCache = new CompactExpertCache(planeOptions, createRoutedExpertCounters());
      const abort = new AbortController(); abortPlane = () => abort.abort(Error('cancel authenticated planes'));
      const beforeAbort = state.writes;
      await assert.rejects(planeCache.prepare(plan, [7], abort.signal), /cancel authenticated planes/);
      assert.equal(state.writes, beforeAbort); assert(partLeases.every(lease => lease.bytes.byteLength === 0));
      abortPlane = null;
      verify(await planeCache.prepare(plan, [7], new AbortController().signal), [7]);
    } finally { state.failAt = null; await planeCache?.destroy(); await planeWorker.close(); }
    state.calls.length = 0;
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    const larger = [7, 2, 5, 0, 1, 3, 6];
    verify(await cache.prepare(plan, larger, new AbortController().signal), larger);
    assert.deepEqual(state.calls.sort((a, b) => b - a), [9, 9, 3], 'larger miss sets retain bounded maximum grouping');
    await cache.destroy();
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    state.failAt = state.writes + 5;
    await assert.rejects(cache.prepare(plan, ids, new AbortController().signal), /grouped upload/);
    assert.equal(state.leases, 0);
    state.failAt = null;
    verify(await cache.prepare(plan, ids, new AbortController().signal), ids);
    await cache.destroy();
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    const abort = new AbortController(); state.abortAfterRead = () => abort.abort();
    await assert.rejects(cache.prepare(plan, ids, abort.signal), /abort/i);
    assert.equal(state.leases, 0); state.abortAfterRead = null;
    verify(await cache.prepare(plan, ids, new AbortController().signal), ids);
    await cache.destroy();
    state.calls.length = 0;
    cache = new CompactExpertCache({ ...options, rangeLease: { ...rangeLease, maxPartBytes: 48 } }, createRoutedExpertCounters());
    verify(await cache.prepare(plan, ids, new AbortController().signal), ids);
    assert.deepEqual(state.calls, [6, 6], 'an admitted smaller output derives two experts per job');
    await cache.destroy();
    for (const change of [{ maxRanges: undefined }, { maxRanges: 3 }]) {
      const rejected = new CompactExpertCache({ ...options, rangeLease: { ...rangeLease, ...change } }, createRoutedExpertCounters());
      await assert.rejects(rejected.prepare(plan, ids, new AbortController().signal), /explicitly admitted maxRanges/);
      await rejected.destroy();
    }
    for (const value of [null, 0, 1.5, 33]) assert.throws(() => validateQwen35RoutedConfig({ ...policy, expertBatchSize: value }), /expertBatchSize/);
  } finally { await cache?.destroy(); await worker?.close(); await rm(root, { recursive: true, force: true }); }
});
