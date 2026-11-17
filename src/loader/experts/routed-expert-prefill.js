import { readRoutedExpertPayload, readRoutedExpertRange, recordRoutedExpertSourceTimings } from './routed-expert-source.js';

/** @type {import('./routed-expert-prefill.js').writeRoutedPrefillExpert} */
export async function writeRoutedPrefillExpert(options, stats, plan, expert, signal, working, scratch, beforeUpload) {
  /** @param {import('./routed-expert-plan.js').RoutedExpertPlane} plane
   * @param {number} offset @param {Uint8Array} bytes */
  const upload = (plane, offset, bytes) => {
    signal.throwIfAborted();
    const target = working.get(plane.role);
    if (!target) throw new Error('Direct prefill requires a complete working bank.');
    beforeUpload();
    const start = performance.now();
    options.device.queue.writeBuffer(target, expert * plane.expertBytes + offset, bytes);
    stats.uploadEnqueueMs += performance.now() - start;
    stats.uploadBytes += bytes.byteLength;
  };
  const start = performance.now();
  try {
    if (options.policy.sourceReadMode === 'expert-parallel') {
      if (!scratch) throw new Error('Parallel prefill requires an owned complete expert scratch.');
      await readRoutedExpertPayload(options, stats, plan, expert, signal, scratch, null);
      let offset = 0;
      for (const plane of plan.planes) {
        upload(plane, 0, scratch.subarray(offset, offset + plane.expertBytes));
        offset += plane.expertBytes;
      }
      return;
    }
    for (const plane of plan.planes) {
      const location = plane.location;
      const physical = location.spans?.length === 1 ? location.spans[0] : location;
      const port = options.rangeLease;
      const leased = port && location.sourceTransform == null && (!location.spans || location.spans.length <= 1);
      if (leased && (typeof physical.shardIndex !== 'number' || !Number.isSafeInteger(physical.shardIndex)
        || physical.shardIndex < 0 || typeof physical.offset !== 'number' || !Number.isSafeInteger(physical.offset)
        || physical.offset < 0 || !Number.isSafeInteger(physical.size) || physical.size < (expert + 1) * plane.expertBytes)) {
        throw new Error('Owned prefill range exceeds its declared physical plane.');
      }
      for (let offset = 0; offset < plane.expertBytes;) {
        signal.throwIfAborted();
        const chunk = leased ? Math.floor(port.maxPartBytes / 4) * 4 : scratch?.byteLength;
        if (!chunk) throw new Error('Direct prefill requires admitted lease or owned fallback scratch.');
        const length = Math.min(chunk, plane.expertBytes - offset);
        let lease = null;
        try {
          const readStart = performance.now();
          let bytes;
          if (leased) {
            if (typeof physical.shardIndex !== 'number' || typeof physical.offset !== 'number') throw new Error('Missing prefill physical range.');
            lease = await port.read(physical.shardIndex, physical.offset + expert * plane.expertBytes + offset, length, signal);
            if (lease.bytes.byteLength !== length) throw new Error('Owned prefill range length mismatch.');
            signal.throwIfAborted();
            recordRoutedExpertSourceTimings(stats, [lease]);
            bytes = lease.bytes;
            stats.sourceReads++;
            stats.sourceReadBytes += length;
            stats.sourceReadWaitMs += performance.now() - readStart;
          } else {
            bytes = await readRoutedExpertRange(options, stats, plane, expert, offset, length, signal,
              options.readRangeInto && location.sourceTransform == null && scratch ? scratch.subarray(0, length) : null);
          }
          upload(plane, offset, bytes);
        } finally {
          // writeBuffer copied the view synchronously; no queued command borrows it.
          lease?.release();
        }
        offset += length;
      }
    }
  } finally { stats.ioWaitMs += performance.now() - start; }
}

/** @type {import('./routed-expert-prefill.js').selectPrefillSeedExperts} */
export function selectPrefillSeedExperts(ids, capacity) {
  const observed = new Map();
  for (const [position, expert] of ids.entries()) {
    const previous = observed.get(expert);
    observed.set(expert, { count: (previous?.count ?? 0) + 1, position });
  }
  return [...observed.keys()].sort((a, b) => observed.get(b).count - observed.get(a).count
    || observed.get(b).position - observed.get(a).position).slice(0, capacity);
}
