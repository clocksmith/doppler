import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CompactExpertCache } from '../../src/loader/experts/compact-expert-cache.js';
import { createRoutedExpertCounters } from '../../src/loader/experts/routed-expert-source.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { test } from 'node:test';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
test('bounded compact plane scheduling and settled sibling failures', { timeout: 5000 }, async () => {
  const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
  const encode = value => new TextEncoder().encode(JSON.stringify(value));
  function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }
  const directory = await mkdtemp(join(tmpdir(), 'compact-plane-'));
  const filename = join(directory, 'weights.bin');
  const bytes = Uint8Array.from({ length: 96 }, (_, i) => i + 1);
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: bytes.length }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 8,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: Array.from({ length: 12 }, (_, i) => ({ offset: i * 8,
      size: 8, identity: sha(bytes.subarray(i * 8, i * 8 + 8)) })) }] });
  const plan = { layer: 0, numLayers: 1, numExperts: 4, expertBytes: 24, intermediateSize: 2,
    planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: role, expertBytes: 8,
      location: { shardIndex: 0, offset: i * 32, size: 32, shape: [4, 2, 1] } })) };
  const policy = { enabled: true, storageMode: 'compact-decode', evictionPolicy: 'lru', compactLayerBytes: null,
    compactPrefillBytes: 0, compactPrefillReuse: 'none', prefillCompactAdmission: 'none', sourceReadMode: 'lease-parallel',
    prefillUploadMode: 'expert-copy', cacheBytes: 64, hostCacheBytes: 0, hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all',
    stagingBytes: 128, readConcurrency: 4, routeChunkTokens: 2, gpuBudgetBytes: 4096, reserveBytes: 16 };
  let worker, cache, entryGate = null, held = null;
  const published = [], leases = [], buffers = [];
  let started = 0, active = 0, peak = 0, failUpload = false;
  let pendingSibling = null, failedUpload = null, failureGate = null;
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    createBuffer: ({ size, label }) => { const buffer = { size, label, data: new Uint8Array(size), destroy() {} }; buffers.push(buffer); return buffer; },
    queue: { async onSubmittedWorkDone() {}, writeBuffer(buffer, offset, data) {
      if (failUpload && buffer.label.endsWith('_gate')) {
        failUpload = false; failedUpload?.resolve(); throw new Error('Plane upload failure');
      }
      if (buffer.label.endsWith('_source_map')) published.push([...new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4)]);
      buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    } } };
  try {
    await writeFile(filename, bytes);
    worker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      files: { 'weights.bin': filename }, maxPartBytes: 8, memoryBudgetBytes: 96, concurrency: 4 });
    const options = { device, policy, compactLayerCount: 1, otherGpuBytes: () => 0,
      async readRange() { throw new Error('Generic read is forbidden in the leased mode'); },
      rangeLease: { ...worker.rangeLease, async read(shard, offset, length, signal) {
        started++; active++; peak = Math.max(peak, active);
        try {
          if (entryGate) await entryGate.promise;
          const lease = await worker.rangeLease.read(shard, offset, length, signal); leases.push(lease);
          if (failureGate && offset < 32) await failureGate.promise;
          if (held && offset >= 32) { pendingSibling?.resolve(); await held.promise; }
          return { bytes: lease.bytes, release() { lease.release(); active--; } };
        } catch (error) { active--; throw error; }
      } } };
    assert.equal(resolveRuntimeConfig({ loading: { expertCache: { qwen35Routed: policy } } }).loading.expertCache.qwen35Routed.sourceReadMode,
      'lease-parallel', 'The explicit mode survives normal runtime resolution');
    const verify = (weights, expert) => {
      const slot = new Uint32Array(weights.sourceExpertToSlot.data.buffer)[expert];
      for (const [i, role] of ['gate', 'up', 'down'].entries()) {
        assert.deepEqual([...weights[role].buffer.data.subarray(slot * 8, slot * 8 + 8)], [...bytes.subarray(i * 32 + expert * 8, i * 32 + expert * 8 + 8)]);
      }
    };
    const summaries = [];
    for (const [mode, admitted] of [['plane-serial', 1], ['lease-parallel', 3]]) {
      const stats = createRoutedExpertCounters();
      cache = new CompactExpertCache({ ...options, policy: { ...policy, sourceReadMode: mode } }, stats);
      started = 0; peak = 0; entryGate = deferred();
      const preparation = cache.prepare(plan, [2], new AbortController().signal);
      assert.equal(started, admitted, 'A single miss dispatches the declared independent plane tasks before any read completes');
      assert.equal(published.length, summaries.length * 2, 'Source map cannot publish ahead of plane completion');
      entryGate.resolve(); entryGate = null;
      verify(await preparation, 2);
      const reads = started;
      verify(await cache.prepare(plan, [2], new AbortController().signal), 2);
      assert.equal(started, reads, 'Resident repeated decode performs no source reread');
      assert.equal(active, 0);
      assert.equal(stats.sourceReadBytes, 24);
      assert.equal(stats.stagingAllocations, 0);
      assert(stats.peakStagingBytes <= policy.stagingBytes);
      assert(leases.every(lease => lease.bytes.byteLength === 0));
      summaries.push({ mode, singleMissConcurrentPlanes: peak, bytes: stats.sourceReadBytes, repeatReads: started - reads });
      await cache.destroy();
    }
    const boundedStats = createRoutedExpertCounters();
    cache = new CompactExpertCache(options, boundedStats);
    peak = 0; started = 0; entryGate = deferred();
    const bounded = cache.prepare(plan, [0, 1], new AbortController().signal);
    assert.equal(started, 4, 'Two misses share the four admitted lanes rather than opening six leases');
    entryGate.resolve(); entryGate = null;
    const boundedWeights = await bounded;
    verify(boundedWeights, 0); verify(boundedWeights, 1);
    assert.equal(peak, 4); assert.equal(active, 0);
    assert.equal(boundedStats.sourceReadBytes, 48);
    assert.equal(boundedStats.cacheMisses, 2);
    assert(boundedStats.peakStagingBytes <= policy.stagingBytes);
    await cache.destroy();
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    held = deferred(); pendingSibling = deferred(); failedUpload = deferred(); failureGate = deferred(); failUpload = true;
    const beforeFailure = published.length;
    const failing = cache.prepare(plan, [0], new AbortController().signal);
    const expectedFailure = assert.rejects(failing, /Plane upload failure/);
    let settled = false; failing.then(() => { settled = true; }, () => { settled = true; });
    // Hold gate upload until a sibling actually owns a lease. Otherwise its early
    // failure can cancel the other acquisitions before this event ever occurs.
    await pendingSibling.promise;
    failureGate.resolve();
    await failedUpload.promise;
    await Promise.resolve();
    assert.equal(settled, false, 'Failure waits for all already admitted siblings');
    assert.equal(published.length, beforeFailure);
    held.resolve(); held = null;
    await expectedFailure;
    failureGate = null;
    assert.equal(active, 0);
    assert(leases.every(lease => lease.bytes.byteLength === 0));
    verify(await cache.prepare(plan, [0], new AbortController().signal), 0);
    await cache.destroy();
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    held = deferred(); pendingSibling = deferred();
    const abort = new AbortController(), beforeCancel = published.length;
    const cancelled = cache.prepare(plan, [1], abort.signal);
    const expectedCancel = assert.rejects(cancelled, /Cancelled parallel planes/);
    let cancelledSettled = false;
    cancelled.then(() => { cancelledSettled = true; }, () => { cancelledSettled = true; });
    await pendingSibling.promise;
    abort.abort(new Error('Cancelled parallel planes'));
    await Promise.resolve();
    assert.equal(cancelledSettled, false, 'Cancellation retains sibling ownership until the actual reads settle');
    assert.equal(published.length, beforeCancel);
    held.resolve(); held = null;
    await expectedCancel;
    assert.equal(active, 0);
    assert(leases.every(lease => lease.bytes.byteLength === 0));
    verify(await cache.prepare(plan, [1], new AbortController().signal), 1);
    await cache.destroy();
    const beforeInvalid = buffers.length;
    cache = new CompactExpertCache({ ...options, rangeLease: undefined }, createRoutedExpertCounters());
    await assert.rejects(cache.prepare(plan, [0], new AbortController().signal), /owned range port/);
    assert.equal(buffers.length, beforeInvalid, 'Missing lease admission fails before GPU allocation');
    const transformed = { ...plan, planes: plan.planes.map(plane => ({ ...plane, location: { ...plane.location, sourceTransform: {} } })) };
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    await assert.rejects(cache.prepare(transformed, [0], new AbortController().signal), /unchanged single-span/);
    assert.equal(buffers.length, beforeInvalid);
    console.log('compact plane scheduling CPU evidence: ' + JSON.stringify(summaries));
  } finally {
    entryGate?.resolve(); held?.resolve(); failureGate?.resolve();
    await cache?.destroy(); await worker?.close(); await rm(directory, { recursive: true });
  }
});
