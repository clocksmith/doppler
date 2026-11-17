import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../config/schema/runtime/loading.schema.js';
import { loadTensorRange } from '../tensors/tensor-reader.js';
import { snapshotVerifiedRangeLeaseTimings } from '../../storage/verified-piece-timings.js';

export function createRoutedExpertCounters() {
  return { prefillSeedBytes: 0, prefillSeedSubmissions: 0, prefillGpuReuseBytes: 0, prefillGpuReuseExperts: 0, prefillGpuReuseSubmissions: 0, sourceReadIntoBytes: 0, stagingAllocations: 0, stagingReuses: 0, sourceReadBytes: 0, sourceReads: 0, uploadBytes: 0,
    cacheHits: 0, cacheMisses: 0, workingHits: 0, workingBufferAllocations: 0, workingBufferAllocatedBytes: 0, workingBufferReuses: 0, workingBufferReusedBytes: 0, evictions: 0, evictionFenceWaits: 0, evictionFenceSkips: 0, bufferReuses: 0, expertCopySubmissions: 0, hitBatchSubmissions: 0, batchedHitExperts: 0,
    residentBytes: 0, peakResidentBytes: 0, peakAccountedGpuBytes: 0, peakStagingBytes: 0,
    ioWaitMs: 0, sourceReadWaitMs: 0, uploadEnqueueMs: 0, initialQueueIdleForWeightsMs: 0,
    sourceTimingJobs: 0, sourceUntimedJobs: 0, sourceJobWallMs: 0, sourceJobReadWaitMs: 0,
    sourceJobVerificationMs: 0, sourceJobCopyMs: 0, sourceJobRoundTripMs: 0,
    hostPrefillAdmissionSkips: 0, hostPrefillAdmissions: 0, hostDecodeAdmissions: 0, hostCacheHits: 0, hostCacheMisses: 0, hostEvictions: 0, hostAdmissionSkips: 0, peakHostOwnedOrReservedBytes: 0 };
}

/** @type {typeof import('./routed-expert-source.js').recordRoutedExpertSourceTimings} */
export function recordRoutedExpertSourceTimings(stats, leases) {
  // Validate the entire successful acquisition before changing coverage or sums.
  // Missing metadata is unsupported, never a fabricated zero-duration job.
  const timings = leases.map(lease => snapshotVerifiedRangeLeaseTimings(lease?.timings));
  let jobs = 0, untimed = 0, wall = 0, read = 0, verification = 0, copy = 0, roundTrip = 0;
  for (const timing of timings) {
    if (!timing) { untimed++; continue; }
    jobs++; wall += timing.workerWallMs; read += timing.readWaitMs;
    verification += timing.verificationMs; copy += timing.copyMs; roundTrip += timing.roundTripMs;
  }
  stats.sourceTimingJobs += jobs;
  stats.sourceUntimedJobs += untimed;
  stats.sourceJobWallMs += wall;
  stats.sourceJobReadWaitMs += read;
  stats.sourceJobVerificationMs += verification;
  stats.sourceJobCopyMs += copy;
  stats.sourceJobRoundTripMs += roundTrip;
}

/** @type {typeof import('./routed-expert-source.js').readRoutedExpertRange} */
export async function readRoutedExpertRange(options, stats, plane, expert, offset, length, signal, destination = null) {
  const readRangeInto = options.readRangeInto;
  return loadTensorRange(plane.location, plane.name, expert * plane.expertBytes + offset, length,
  async (shard, start, size) => {
    signal.throwIfAborted();
    const readStarted = performance.now();
    const data = await options.readRange(shard, start, size, signal);
    stats.sourceReadWaitMs += performance.now() - readStarted;
    if (data.byteLength !== size) throw new Error('Routed expert source must return exactly the requested byte range.');
    signal.throwIfAborted();
    recordRoutedExpertSourceTimings(stats, [null]);
    stats.sourceReads++;
    stats.sourceReadBytes += data.byteLength;
    return data;
  }, readRangeInto ? async (shard, start, destination) => {
    signal.throwIfAborted();
    const readStarted = performance.now();
    const count = await readRangeInto(shard, start, destination, signal);
    stats.sourceReadWaitMs += performance.now() - readStarted;
    if (count !== destination.byteLength) throw new Error('Routed expert source must fill exactly the requested destination.');
    signal.throwIfAborted();
    recordRoutedExpertSourceTimings(stats, [null]);
    stats.sourceReads++;
    stats.sourceReadBytes += count;
    stats.sourceReadIntoBytes += count;
    return count;
  } : null, destination);
}

/** @param {import('./routed-expert-cache.js').RoutedExpertCacheOptions} options
 * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan */
export function assertExpertReadShape(options, plan) {
  if (options.policy.sourceReadMode === 'lease-parallel' || options.policy.sourceReadMode === 'expert-lease-batch') {
    if (!options.rangeLease || plan.planes.some(plane => plane.location.sourceTransform != null
      || (plane.location.spans && plane.location.spans.length !== 1))) {
      throw new Error('Parallel leased planes require an owned range port and unchanged single-span planes.');
    }
    if (options.policy.sourceReadMode === 'expert-lease-batch'
      && (typeof options.rangeLease.readMany !== 'function' || plan.planes.length < 1 || plan.planes.length > 3
        || plan.expertBytes > options.rangeLease.maxPartBytes)) {
      throw new Error('Batched expert leases require one admitted output covering all unchanged expert planes.');
    }
    if (options.policy.sourceReadMode === 'expert-lease-batch') {
      const size = options.policy.expertBatchSize ?? DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed.expertBatchSize;
      if (size > 1 && (!Number.isSafeInteger(options.rangeLease.maxRanges)
        || Number(options.rangeLease.maxRanges) < size * plan.planes.length)) {
        throw new Error('Multi-expert acquisition requires an explicitly admitted maxRanges for every declared plane.');
      }
    }
    return;
  }
  if (options.policy.sourceReadMode !== 'expert-parallel') return;
  const limit = Math.floor(options.policy.stagingBytes / (2 * options.policy.readConcurrency * 4)) * 4;
  if (options.rangeLease || !options.readRangeInto || plan.expertBytes > limit
    || plan.planes.some(plane => plane.location.sourceTransform != null)) {
    throw new Error('Parallel expert reads require direct ranges, no lease port, unchanged packed planes and one complete bounded scratch payload.');
  }
}

/** @type {typeof import('./routed-expert-source.js').readRoutedExpertLease} */
export async function readRoutedExpertLease(options, stats, plan, expert, signal, layerStats) {
  return readRoutedExpertsLease(options, stats, plan, [expert], signal, layerStats);
}

/** @param {import('./routed-expert-cache.js').RoutedExpertCacheOptions} options
 * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
 * @param {number[]} experts */
function prepareExpertLeaseParts(options, plan, experts) {
  assertExpertReadShape(options, plan);
  const port = options.rangeLease;
  if (options.policy.sourceReadMode !== 'expert-lease-batch' || !port?.readMany
    || !experts.length || experts.length > (options.policy.expertBatchSize ?? DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed.expertBatchSize)
    || new Set(experts).size !== experts.length
    || experts.some(expert => !Number.isSafeInteger(expert) || expert < 0 || expert >= plan.numExperts)
    || experts.length * plan.planes.length > (port?.maxRanges ?? 3)
    || experts.length * plan.expertBytes > (port?.maxPartBytes ?? 0)
    || plan.planes.reduce((sum, plane) => sum + plane.expertBytes, 0) !== plan.expertBytes) {
    throw new Error('Batched expert lease requires an exact declared expert extent.');
  }
  const parts = experts.flatMap(expert => plan.planes.map(plane => {
    const physical = plane.location.spans?.[0] ?? plane.location;
    const offset = Number(physical.offset) + expert * plane.expertBytes;
    if (!Number.isSafeInteger(physical.shardIndex) || Number(physical.shardIndex) < 0
      || !Number.isSafeInteger(physical.offset) || Number(physical.offset) < 0
      || !Number.isSafeInteger(physical.size) || Number(physical.size) < 0
      || !Number.isSafeInteger(Number(physical.offset) + Number(physical.size))
      || !Number.isSafeInteger(plane.expertBytes) || plane.expertBytes <= 0 || plane.expertBytes % 4
      || !Number.isSafeInteger(offset) || offset % 4 || !Number.isSafeInteger(offset + plane.expertBytes)
      || offset + plane.expertBytes > Number(physical.offset) + Number(physical.size)) {
      throw new Error('Batched expert range exceeds its declared physical plane.');
    }
    return { shard: Number(physical.shardIndex), offset, length: plane.expertBytes };
  }));
  return { port, parts, readMany: port.readMany.bind(port) };
}

/** @type {typeof import('./routed-expert-source.js').readRoutedExpertsLease} */
export async function readRoutedExpertsLease(options, stats, plan, experts, signal, layerStats) {
  const { parts, readMany } = prepareExpertLeaseParts(options, plan, experts);
  signal.throwIfAborted();
  const started = performance.now();
  const lease = await readMany(parts, signal);
  try {
    signal.throwIfAborted();
    if (lease.bytes.byteLength !== plan.expertBytes * experts.length) throw new Error('Batched expert range length mismatch.');
    recordRoutedExpertSourceTimings(stats, [lease]);
    const elapsed = performance.now() - started;
    stats.sourceReads += parts.length;
    stats.sourceReadBytes += plan.expertBytes * experts.length;
    stats.sourceReadWaitMs += elapsed;
    if (layerStats) {
      layerStats.sourceReadBytes += plan.expertBytes * experts.length;
      layerStats.sourceReadWaitMs += elapsed;
    }
    return lease;
  } catch (error) { lease.release(); throw error; }
}

/** @type {typeof import('./routed-expert-source.js').readRoutedExpertPlaneLeases} */
export async function readRoutedExpertPlaneLeases(options, stats, plan, expert, signal, layerStats) {
  const { port, parts, readMany } = prepareExpertLeaseParts(options, plan, [expert]);
  if (!Number.isSafeInteger(port.concurrency) || Number(port.concurrency) < parts.length) {
    throw new Error('Parallel plane leases require one admitted worker lane per expert plane.');
  }
  signal.throwIfAborted();
  const started = performance.now();
  // Each owned plane goes to a separate lane. Do not return any successful
  // prefix until all native reads, authentication and failures have settled.
  const outcomes = await Promise.allSettled(parts.map(async part => readMany([part], signal)));
  const leases = outcomes.flatMap(outcome => outcome.status === 'fulfilled' ? [outcome.value] : []);
  try {
    signal.throwIfAborted();
    const failed = outcomes.find(outcome => outcome.status === 'rejected');
    if (failed) throw failed.reason;
    if (leases.some((lease, index) => lease.bytes.byteLength !== parts[index].length)) {
      throw new Error('Parallel expert plane range length mismatch.');
    }
    recordRoutedExpertSourceTimings(stats, leases);
    const elapsed = performance.now() - started;
    stats.sourceReads += parts.length;
    stats.sourceReadBytes += plan.expertBytes;
    stats.sourceReadWaitMs += elapsed;
    if (layerStats) {
      layerStats.sourceReadBytes += plan.expertBytes;
      layerStats.sourceReadWaitMs += elapsed;
    }
    return leases;
  } catch (error) {
    const failures = [];
    for (const lease of leases) {
      try { lease.release(); } catch (failure) { failures.push(failure); }
    }
    if (failures.length) throw new AggregateError([error, ...failures], 'Parallel plane acquisition and lease cleanup failed.');
    throw error;
  }
}

/** @param {import('./routed-expert-cache.js').RoutedExpertCacheOptions} options
 * @param {import('./routed-expert-source.js').RoutedExpertCounters} stats
 * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
 * @param {number} expert @param {AbortSignal} signal @param {Uint8Array} destination
 * @param {import('./routed-expert-cache.js').CompactLayerStats | null} layerStats */
export async function readRoutedExpertPayload(options, stats, plan, expert, signal, destination, layerStats) {
  assertExpertReadShape(options, plan);
  signal.throwIfAborted();
  if (options.policy.sourceReadMode !== 'expert-parallel'
    || !(destination instanceof Uint8Array) || destination.byteLength !== plan.expertBytes
    || plan.planes.reduce((sum, plane) => sum + plane.expertBytes, 0) !== destination.byteLength) {
    throw new Error('Expert scratch must cover exactly the declared unchanged planes in expert-parallel mode.');
  }
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal.addEventListener('abort', abort, { once: true });
  if (signal.aborted) abort();
  let offset = 0;
  try {
    const reads = plan.planes.map(plane => {
      const view = destination.subarray(offset, offset + plane.expertBytes);
      offset += plane.expertBytes;
      const started = layerStats ? performance.now() : 0;
      return readRoutedExpertRange(options, stats, plane, expert, 0, plane.expertBytes,
        controller.signal, view).then(bytes => {
          if (layerStats) {
            layerStats.sourceReadBytes += bytes.byteLength;
            layerStats.sourceReadWaitMs += performance.now() - started;
          }
          return bytes;
        }).catch(error => { controller.abort(error); throw error; });
    });
    // Keep each disjoint view owned until all sources settle, including aborts.
    const outcomes = await Promise.allSettled(reads);
    const errors = [...new Set(outcomes.filter(r => r.status === 'rejected').map(r => r.reason))];
    if (errors.length > 1) throw new AggregateError(errors, 'Parallel expert acquisition failed.');
    if (errors.length) throw errors[0];
    signal.throwIfAborted();
    return destination;
  } finally { signal.removeEventListener('abort', abort); }
}
