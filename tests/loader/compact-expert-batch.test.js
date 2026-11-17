import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CompactExpertCache } from '../../src/loader/experts/compact-expert-cache.js';
import { createRoutedExpertCounters } from '../../src/loader/experts/routed-expert-source.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
test('one authenticated owned batch per expert, upload lifetime, failure and recovery', { timeout: 5000 }, async () => {
  const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
  const encode = value => new TextEncoder().encode(JSON.stringify(value));
  const directory = await mkdtemp(join(tmpdir(), 'compact-batch-'));
  const bytes = Uint8Array.from({ length: 96 }, (_, i) => i + 1);
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: bytes.length }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 8,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: Array.from({ length: 12 }, (_, i) => ({
      offset: i * 8, size: 8, identity: sha(bytes.subarray(i * 8, i * 8 + 8)),
    })) }] });
  const plan = { layer: 0, numLayers: 1, numExperts: 4, expertBytes: 24, intermediateSize: 2,
    planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: role, expertBytes: 8,
      location: { shardIndex: 0, offset: i * 32, size: 32, shape: [4, 2, 1] } })) };
  const policy = { enabled: true, storageMode: 'compact-decode', evictionPolicy: 'lru', compactLayerBytes: null,
    compactPrefillBytes: 0, compactPrefillReuse: 'none', prefillCompactAdmission: 'none', sourceReadMode: 'expert-lease-batch',
    prefillUploadMode: 'expert-copy', cacheBytes: 160, hostCacheBytes: 0, hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all',
    stagingBytes: 128, readConcurrency: 2, routeChunkTokens: 2, gpuBudgetBytes: 4096, reserveBytes: 16 };
  let worker, cache, requests = 0, active = 0, uploads = 0, failUploadAt = null, afterRead = null;
  const leases = [], allocations = [];
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    createBuffer(description) {
      const value = { ...description, data: new Uint8Array(description.size), destroyed: false,
        destroy() { this.destroyed = true; } };
      allocations.push(value); return value;
    },
    queue: { async onSubmittedWorkDone() {}, writeBuffer(target, offset, data) {
      uploads++;
      assert(data.byteLength > 0, 'Borrowed source stays attached through every upload');
      if (uploads === failUploadAt) throw Error('injected second-plane upload failure');
      target.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    } } };
  try {
    await writeFile(join(directory, 'weights.bin'), bytes);
    worker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      files: { 'weights.bin': join(directory, 'weights.bin') }, maxPartBytes: 24, memoryBudgetBytes: 80, concurrency: 2 });
    const port = { ...worker.rangeLease, async read() { throw Error('Per-plane read must not run in explicit batch mode'); },
      async readMany(parts, signal) {
        requests++; active++;
        try {
          const lease = await worker.rangeLease.readMany(parts, signal); leases.push(lease);
          afterRead?.();
          return { bytes: lease.bytes, release() { lease.release(); active--; } };
        } catch (error) { active--; throw error; }
      } };
    const options = { device, policy, compactLayerCount: 1, otherGpuBytes: () => 0, rangeLease: port,
      async readRange() { throw Error('Generic fallback is not admitted in batch mode'); } };
    assert.equal(resolveRuntimeConfig({ loading: { expertCache: { qwen35Routed: policy } } }).loading.expertCache.qwen35Routed.sourceReadMode,
      'expert-lease-batch');
    const verify = (weights, expert) => {
      const slot = new Uint32Array(weights.sourceExpertToSlot.data.buffer)[expert];
      for (const [i, role] of ['gate', 'up', 'down'].entries()) {
        assert.deepEqual(weights[role].buffer.data.slice(slot * 8, slot * 8 + 8), bytes.slice(i * 32 + expert * 8, i * 32 + expert * 8 + 8));
      }
    };
    const stats = createRoutedExpertCounters();
    cache = new CompactExpertCache(options, stats);
    const result = await cache.prepare(plan, [3, 1], new AbortController().signal);
    verify(result, 3); verify(result, 1);
    assert.equal(requests, 2, 'Three planes share one read request per miss');
    assert.equal(stats.sourceReads, 6, 'Logical plane byte accounting remains comparable');
    assert.equal(stats.sourceReadBytes, 48);
    assert.equal(stats.stagingAllocations, 0, 'No main-side payload scratch or gather allocation');
    assert.equal(active, 0);
    assert(leases.every(lease => lease.bytes.byteLength === 0), 'One release detaches all sibling plane views');
    await cache.prepare(plan, [3, 1], new AbortController().signal);
    assert.equal(requests, 2, 'Resident experts keep hot reuse');
    await cache.destroy();

    for (const change of [{ readMany: undefined }, { maxPartBytes: 16 }]) {
      const before = allocations.length;
      cache = new CompactExpertCache({ ...options, rangeLease: { ...port, ...change } }, createRoutedExpertCounters());
      await assert.rejects(cache.prepare(plan, [0], new AbortController().signal), /one admitted output/);
      assert.equal(allocations.length, before, 'Invalid batch capability rejects before GPU allocation');
      await cache.destroy();
    }
    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    failUploadAt = uploads + 2;
    await assert.rejects(cache.prepare(plan, [0], new AbortController().signal), /second-plane/);
    assert.equal(active, 0);
    assert(leases.every(lease => lease.bytes.byteLength === 0));
    failUploadAt = null;
    verify(await cache.prepare(plan, [0], new AbortController().signal), 0);
    const abort = new AbortController();
    afterRead = () => abort.abort(Error('abort before batch upload'));
    await assert.rejects(cache.prepare(plan, [2], abort.signal), /abort before batch/);
    assert.equal(active, 0);
    assert(leases.every(lease => lease.bytes.byteLength === 0));
    afterRead = null;
    verify(await cache.prepare(plan, [2], new AbortController().signal), 2);
    await cache.destroy();

    cache = new CompactExpertCache(options, createRoutedExpertCounters());
    const corrupt = bytes.slice(); corrupt[88] ^= 128;
    await writeFile(join(directory, 'weights.bin'), corrupt);
    const beforeCorruption = uploads;
    await assert.rejects(cache.prepare(plan, [3], new AbortController().signal), /integrity mismatch/);
    assert.equal(uploads, beforeCorruption, 'A corrupt last plane prevents every plane upload and map publication');
    assert.equal(active, 0);
    await writeFile(join(directory, 'weights.bin'), bytes);
    verify(await cache.prepare(plan, [3], new AbortController().signal), 3);
    await cache.destroy();
    assert(allocations.every(value => value.destroyed), 'Every owned model bank retires after failure/recovery');
  } finally { await cache?.destroy(); await worker?.close(); await rm(directory, { recursive: true }); }
});
