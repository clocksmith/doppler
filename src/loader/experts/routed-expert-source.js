import { loadTensorRange } from '../tensors/tensor-reader.js';

export function createRoutedExpertCounters() {
  return { sourceReadIntoBytes: 0, stagingAllocations: 0, stagingReuses: 0, sourceReadBytes: 0, sourceReads: 0, uploadBytes: 0,
    cacheHits: 0, cacheMisses: 0, workingHits: 0, workingBufferAllocations: 0, workingBufferAllocatedBytes: 0, workingBufferReuses: 0, workingBufferReusedBytes: 0, evictions: 0, evictionFenceWaits: 0, evictionFenceSkips: 0, bufferReuses: 0, expertCopySubmissions: 0, hitBatchSubmissions: 0, batchedHitExperts: 0,
    residentBytes: 0, peakResidentBytes: 0, peakAccountedGpuBytes: 0, peakStagingBytes: 0,
    ioWaitMs: 0, sourceReadWaitMs: 0, uploadEnqueueMs: 0, initialQueueIdleForWeightsMs: 0,
    hostPrefillAdmissionSkips: 0, hostPrefillAdmissions: 0, hostDecodeAdmissions: 0, hostCacheHits: 0, hostCacheMisses: 0, hostEvictions: 0, hostAdmissionSkips: 0, peakHostOwnedOrReservedBytes: 0 };
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
    stats.sourceReads++;
    stats.sourceReadBytes += data.byteLength;
    return data;
  }, readRangeInto ? async (shard, start, destination) => {
    signal.throwIfAborted();
    const readStarted = performance.now();
    const count = await readRangeInto(shard, start, destination, signal);
    stats.sourceReadWaitMs += performance.now() - readStarted;
    if (count !== destination.byteLength) throw new Error('Routed expert source must fill exactly the requested destination.');
    stats.sourceReads++;
    stats.sourceReadBytes += count;
    stats.sourceReadIntoBytes += count;
    return count;
  } : null, destination);
}
