import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRoutedExpertCounters, recordRoutedExpertSourceTimings, readRoutedExpertsLease,
  readRoutedExpertPlaneLeases, readRoutedExpertRange } from '../../src/loader/experts/routed-expert-source.js';
import { writeRoutedPrefillExpert } from '../../src/loader/experts/routed-expert-prefill.js';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const timings = Object.freeze({ workerWallMs: 4, readWaitMs: 5, verificationMs: 2, copyMs: 1, roundTripMs: 8 });
const plan = { numExperts: 2, expertBytes: 12, planes: [0, 1, 2].map(shardIndex => ({ role: String(shardIndex),
  expertBytes: 4, name: String(shardIndex), location: { shardIndex, offset: 0, size: 8 } })) };
const signal = () => new AbortController().signal;
const options = readMany => ({ policy: { sourceReadMode: 'expert-lease-batch', expertBatchSize: 1 },
  rangeLease: { concurrency: 3, maxPartBytes: 12, maxRanges: 3, readMany } });
test('source coverage distinguishes unsupported leases and sums validated overlapping observations only', () => {
  const stats = createRoutedExpertCounters(), lease = { bytes: new Uint8Array(4), timings, release() {} };
  recordRoutedExpertSourceTimings(stats, [lease, { ...lease, timings: undefined }, null]);
  assert.equal(stats.sourceTimingJobs, 1); assert.equal(stats.sourceUntimedJobs, 2);
  assert.equal(stats.sourceJobWallMs, 4); assert.equal(stats.sourceJobReadWaitMs, 5);
  assert.equal(stats.sourceJobVerificationMs, 2); assert.equal(stats.sourceJobCopyMs, 1); assert.equal(stats.sourceJobRoundTripMs, 8);
  const before = { ...stats };
  assert.throws(() => recordRoutedExpertSourceTimings(stats, [lease, { ...lease, timings: { ...timings, copyMs: -1 } }]), /timing metadata/);
  assert.deepEqual(stats, before, 'An invalid sibling prevents partial timing/counter publication');
});
test('whole-expert and plane leases count worker jobs rather than logical ranges and release malformed results', async () => {
  const stats = createRoutedExpertCounters(); let released = 0, invalid = false;
  const port = options(parts => ({ bytes: new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0)),
    timings: invalid ? { ...timings, verificationMs: Infinity } : timings, release() { released++; } }));
  (await readRoutedExpertsLease(port, stats, plan, [1], signal(), null)).release();
  assert.equal(stats.sourceTimingJobs, 1); assert.equal(stats.sourceReads, 3);
  const planes = await readRoutedExpertPlaneLeases(port, stats, plan, 1, signal(), null);
  assert.equal(stats.sourceTimingJobs, 4); assert.equal(stats.sourceJobVerificationMs, 8);
  planes.forEach(lease => lease.release());
  const before = { ...stats }; invalid = true;
  await assert.rejects(readRoutedExpertPlaneLeases(port, stats, plan, 1, signal(), null), /timing metadata/);
  assert.equal(released, 7); assert.deepEqual(stats, before);
});
test('generic ranges count unsupported jobs and direct prefill propagates worker scalars before uploads', async () => {
  const stats = createRoutedExpertCounters(); let released = 0, writes = 0, bad = false;
  await readRoutedExpertRange({ readRange: async () => new ArrayBuffer(4) }, stats, plan.planes[0], 1, 0, 4, signal());
  assert.equal(stats.sourceUntimedJobs, 1); assert.equal(stats.sourceTimingJobs, 0);
  const port = { ...options(), device: { queue: { writeBuffer() { writes++; } } }, rangeLease: {
    maxPartBytes: 4, read: async () => ({ bytes: new Uint8Array(4), timings: bad ? {} : timings, release() { released++; } }) } };
  await writeRoutedPrefillExpert(port, stats, plan, 1, signal(), new Map(plan.planes.map(plane => [plane.role, {}])), null, () => {});
  assert.equal(stats.sourceTimingJobs, 3); assert.equal(stats.sourceJobReadWaitMs, 15); assert.equal(writes, 3); assert.equal(released, 3);
  const before = { ...stats }; bad = true;
  await assert.rejects(writeRoutedPrefillExpert(port, stats, plan, 1, signal(),
    new Map(plan.planes.map(plane => [plane.role, {}])), null, () => {}), /timing metadata/);
  assert.equal(writes, 3); assert.equal(released, 4); assert.equal(stats.sourceTimingJobs, before.sourceTimingJobs);
});

test('real worker timings reach separate prefill/decode phase counters without receipt IPC', { timeout: 5000 }, async () => {
  globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
  const prior = getRuntimeConfig(), root = await mkdtemp(join(tmpdir(), 'phase-timings-'));
  const bytes = Uint8Array.from({ length: 96 }, (_, index) => index + 1);
  const encode = value => new TextEncoder().encode(JSON.stringify(value));
  const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 96 }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 8,
    files: [{ path: 'weights.bin', size: 96, pieces: Array.from({ length: 12 }, (_, index) => ({
      offset: index * 8, size: 8, identity: sha(bytes.subarray(index * 8, index * 8 + 8)),
    })) }] });
  const selected = { layer: 0, numLayers: 1, numExperts: 4, expertBytes: 24, bankBytes: 96, intermediateSize: 2,
    planes: ['gate', 'up', 'down'].map((role, index) => ({ role, name: role, dtype: 'f32', expertBytes: 8, bytes: 32,
      location: { shardIndex: 0, offset: index * 32, size: 32, shape: [4, 2, 1] } })) };
  const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
    storageMode: 'compact-decode', sourceReadMode: 'expert-lease-batch', expertBatchSize: 1,
    compactPrefillBytes: 128, compactPrefillReuse: 'resident-gpu', prefillCompactAdmission: 'none', prefillUploadMode: 'direct-bank',
    cacheBytes: 256, stagingBytes: 320, readConcurrency: 4, gpuBudgetBytes: 4096, reserveBytes: 16 };
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    createBuffer(description) { return { ...description, data: new Uint8Array(description.size), destroy() {} }; },
    queue: { async onSubmittedWorkDone() {}, writeBuffer(target, offset, source) {
      target.data.set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength), offset);
    } } };
  let gather, cache;
  try {
    await writeFile(join(root, 'weights.bin'), bytes);
    gather = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      files: { 'weights.bin': join(root, 'weights.bin') }, maxPartBytes: 24, concurrency: 4, memoryBudgetBytes: 160 });
    setRuntimeConfig({ shared: { debug: { profiler: { enabled: true } } } });
    cache = new RoutedExpertCache({ device, policy, compactLayerCount: 1, rangeLease: gather.rangeLease,
      otherGpuBytes: () => 0, readRange: async () => { throw new Error('Generic fallback must not execute'); } });
    await cache.prepare(selected, [0, 1, 2, 3], undefined, 'prefill');
    const prefill = cache.getStats().phaseProfiles.prefill.counters;
    assert.equal(prefill.sourceTimingJobs, 12); assert.equal(prefill.sourceUntimedJobs, 0);
    await cache.prepare(selected, [0, 1], undefined, 'decode');
    const stats = cache.getStats(), decode = stats.phaseProfiles.decode.counters;
    assert.equal(decode.sourceTimingJobs, 2); assert.equal(decode.sourceUntimedJobs, 0);
    assert.equal(stats.sourceTimingJobs, 14); assert.equal(stats.sourceUntimedJobs, 0);
    for (const counter of ['sourceJobWallMs', 'sourceJobReadWaitMs', 'sourceJobVerificationMs', 'sourceJobCopyMs', 'sourceJobRoundTripMs']) {
      assert(Number.isFinite(prefill[counter]) && prefill[counter] >= 0);
      assert(Number.isFinite(decode[counter]) && decode[counter] >= 0);
      assert.equal(stats[counter], prefill[counter] + decode[counter]);
    }
    await cache.prepare(selected, [0, 1], undefined, 'decode');
    assert.equal(cache.getStats().phaseProfiles.decode.counters.sourceTimingJobs, 2, 'Cache hits invent no source observations');
  } finally { await cache?.destroy(); await gather?.close(); setRuntimeConfig(prior); await rm(root, { recursive: true, force: true }); }
});
