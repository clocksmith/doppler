import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CompactExpertCache } from '../../src/loader/experts/compact-expert-cache.js';
import { createRoutedExpertCounters } from '../../src/loader/experts/routed-expert-source.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { ShardCache } from '../../src/loader/shard-cache.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
const sha = b => 'sha256:' + createHash('sha256').update(b).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.from({ length: 16 }, (_, i) => i);
const dir = await mkdtemp(join(tmpdir(), 'compact-gather-'));
const path = join(dir, 'weights.bin');
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 16 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: 16, pieces: [0, 4, 8, 12].map(offset =>
    ({ offset, size: 4, identity: sha(bytes.subarray(offset, offset + 4)) })) }] });
let worker, cache;
try {
  await writeFile(path, bytes);
  worker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    files: { 'weights.bin': path }, maxPartBytes: 8, memoryBudgetBytes: 16 });
  const shards = new ShardCache({ maxEntries: 1 });
  shards.setCustomLoader(null, true, { rangeLease: worker.rangeLease });
  assert.equal(shards.rangeLease, worker.rangeLease);
  shards.setCustomLoader(null);
  assert.equal(shards.rangeLease, null, 'Switching storage clears the old model lease port');
  let failUpload = false;
  const leases = [];
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    createBuffer: ({ size }) => ({ size, data: new Uint8Array(size), destroy() {} }),
    queue: { async onSubmittedWorkDone() {}, writeBuffer(buffer, offset, data) {
      if (failUpload) { failUpload = false; throw new Error('Injected upload failure'); }
      buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    } } };
  const policy = { enabled: true, storageMode: 'compact-decode', evictionPolicy: 'lru', compactLayerBytes: null,
    compactPrefillBytes: 0, compactPrefillReuse: 'none', cacheBytes: 24, hostCacheBytes: 0,
    hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all', stagingBytes: 32, readConcurrency: 2,
    routeChunkTokens: 2, gpuBudgetBytes: 4096, reserveBytes: 16 };
  const plan = { layer: 0, numLayers: 1, numExperts: 2, expertBytes: 8, intermediateSize: 2,
    planes: [{ role: 'down', dtype: 'f32', name: 'down', expertBytes: 8,
      location: { shardIndex: 0, offset: 0, size: 16, shape: [2, 2, 1] } }] };
  const options = { device, policy, compactLayerCount: 1, otherGpuBytes: () => 0,
    async readRange() { throw new Error('Generic read must not run for admitted whole parts'); },
    rangeLease: { ...worker.rangeLease, async read(...args) {
      const lease = await worker.rangeLease.read(...args); leases.push(lease); return lease;
    } } };
  assert.throws(() => new CompactExpertCache({ ...options, policy: { ...policy, stagingBytes: 16 } }, createRoutedExpertCounters()), /aggregate/);
  const stats = createRoutedExpertCounters();
  cache = new CompactExpertCache(options, stats);
  await cache.prepare(plan, [0, 1], new AbortController().signal);
  assert.equal(leases.length, 2);
  assert(leases.every(lease => lease.bytes.byteLength === 0), 'Uploaded ranges returned and detached');
  assert.equal(stats.stagingAllocations, 0, 'No main-thread scratch allocated');
  assert.equal(stats.sourceReadBytes, 16);
  assert(stats.peakStagingBytes <= policy.stagingBytes, 'Worker output and piece buffers counted');
  await cache.destroy();
  cache = new CompactExpertCache(options, createRoutedExpertCounters());
  failUpload = true;
  await assert.rejects(cache.prepare(plan, [0], new AbortController().signal), /upload failure/);
  assert(leases.every(lease => lease.bytes.byteLength === 0), 'Failure returns the owned range');
  const abort = new AbortController();
  const originalRead = options.rangeLease.read;
  options.rangeLease.read = async (...args) => {
    const lease = await originalRead(...args);
    abort.abort(new Error('Cancelled before upload'));
    return lease;
  };
  await assert.rejects(cache.prepare(plan, [0], abort.signal), /Cancelled/);
  assert(leases.every(lease => lease.bytes.byteLength === 0), 'Cancellation releases the authenticated range');
  options.rangeLease.read = originalRead;
  const recovered = await cache.prepare(plan, [0], new AbortController().signal);
  assert.deepEqual([...recovered.down.buffer.data.subarray(0, 8)], [...bytes.subarray(0, 8)]);
  await cache.destroy();
  const beforeFallback = leases.length;
  const fallbackStats = createRoutedExpertCounters();
  const fallbackOptions = { ...options, async readRange(shard, offset, length) {
    assert.equal(shard, 0); return bytes.slice(offset, offset + length).buffer;
  } };
  cache = new CompactExpertCache(fallbackOptions, fallbackStats);
  const spanPlan = { ...plan, planes: [{ ...plan.planes[0], location: {
    ...plan.planes[0].location, spans: [{ shardIndex: 0, offset: 0, size: 8 }, { shardIndex: 0, offset: 8, size: 8 }],
  } }] };
  const fallback = await cache.prepare(spanPlan, [0, 1], new AbortController().signal);
  assert.deepEqual([...fallback.down.buffer.data], [...bytes]);
  assert.equal(leases.length, beforeFallback, 'Multispan source retains the generic range path');
  assert(fallbackStats.peakStagingBytes <= policy.stagingBytes, 'Fallback includes the retained worker budget');
} finally { await cache?.destroy(); await worker?.close(); await rm(dir, { recursive: true }); }
console.log('compact-expert-gather: actual worker upload, aggregate admission, release, failure and retry passed');
