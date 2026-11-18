import { createVerifiedPieceGatherLane } from './verified-piece-gather-lane.js';

/**
 * @param {import('./verified-piece-gather.js').VerifiedPieceProtectedCacheOptions | undefined} rawPolicy
 * @param {number} pieceBytes
 */
async function prepareProtectedCache(rawPolicy, pieceBytes) {
  if (rawPolicy === undefined) return undefined;
  if (!rawPolicy || typeof rawPolicy !== 'object' || Array.isArray(rawPolicy)) {
    throw new Error('Protected piece cache policy must be an explicit object.');
  }
  const { directory, maxBytes } = rawPolicy;
  const minHostAvailableBytes = rawPolicy.minHostAvailableBytes ?? 0;
  const staleAfterMs = rawPolicy.staleAfterMs ?? 90_000;
  const { isAbsolute, join } = await import('node:path');
  const { lstatSync, mkdirSync, readdirSync } = await import('node:fs');
  if (typeof directory !== 'string' || !directory.trim() || !isAbsolute(directory)
    || !Number.isSafeInteger(maxBytes) || maxBytes < pieceBytes
    || !Number.isSafeInteger(minHostAvailableBytes) || minHostAvailableBytes < 0
    || !Number.isSafeInteger(staleAfterMs) || staleAfterMs < 1000) {
    throw new Error('Protected piece cache requires an absolute directory, maxBytes >= maxPieceBytes, and valid bounds.');
  }
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const rootStat = lstatSync(directory);
  const expectedUid = typeof process.geteuid === 'function' ? process.geteuid() : null;
  if (!rootStat.isDirectory() || (expectedUid !== null && rootStat.uid !== expectedUid)) {
    throw new Error('Protected piece cache directory must be an owned non-symlink directory.');
  }
  const rawBuckets = Math.max(1, Math.floor(maxBytes / Math.max(1, 4 * pieceBytes)));
  const bucketCount = Math.max(1, Math.min(256, 1 << Math.floor(Math.log2(rawBuckets))));
  const bucketMaxBytes = Math.floor(maxBytes / bucketCount);
  if (bucketMaxBytes < pieceBytes) {
    throw new Error('Protected piece cache bucket budget must cover at least one complete piece.');
  }
  const sharedBucketBytes = new SharedArrayBuffer(bucketCount * 8);
  const bucketView = new BigInt64Array(sharedBucketBytes);
  for (let index = 0; index < bucketCount; index++) {
    const sub = join(directory, index.toString(16).padStart(2, '0'));
    mkdirSync(sub, { mode: 0o700, recursive: true });
    const subStat = lstatSync(sub);
    if (!subStat.isDirectory() || (expectedUid !== null && subStat.uid !== expectedUid)) {
      throw new Error('Protected piece cache bucket must be an owned non-symlink directory.');
    }
    let bytes = 0n;
    for (const name of readdirSync(sub)) {
      if (!name.endsWith('.piece')) continue;
      try {
        const st = lstatSync(join(sub, name));
        if (st.isFile()) bytes += BigInt(st.size);
      } catch {
        // Concurrent worker cleanup is benign.
      }
    }
    Atomics.store(bucketView, index, bytes);
  }
  return {
    directory,
    maxBytes,
    bucketCount,
    bucketMaxBytes,
    minHostAvailableBytes,
    staleAfterMs,
    sharedBucketBytes,
  };
}

/** @type {import('./verified-piece-gather.js').createVerifiedPieceGather} */
export async function createVerifiedPieceGather(options) {
  options.startupSignal?.throwIfAborted();
  // Omission preserves the original single-lane contract, not automatic tuning.
  const concurrency = options.concurrency ?? 1;
  const maxRanges = options.maxRanges === undefined ? 3 : options.maxRanges;
  if (!Number.isSafeInteger(maxRanges) || maxRanges < 1 || maxRanges > 32) {
    throw new Error('Gather maxRanges must be a positive safe integer no greater than 32.');
  }
  const pinned = { ...options, manifestBytes: new Uint8Array(options.manifestBytes),
    maxRanges,
    indexBytes: new Uint8Array(options.indexBytes), files: { ...options.files },
    ...(options.fileIdentities ? { fileIdentities: structuredClone(options.fileIdentities) } : {}) };
  if (pinned.fileIdentities !== undefined && (!pinned.fileIdentities || typeof pinned.fileIdentities !== 'object'
    || Array.isArray(pinned.fileIdentities) || Object.keys(pinned.fileIdentities).length !== Object.keys(pinned.files).length
    || Object.keys(pinned.files).some(name => {
      const identity = pinned.fileIdentities?.[name];
      const fields = /** @type {const} */ (['device', 'inode', 'size']);
      return !identity || !fields.every(key => typeof identity[key] === 'string' && /^[0-9]+$/.test(identity[key]));
    }))) {
    throw new Error('Gather file identities must bind every approved file device, inode and size.');
  }
  const pieceBytes = JSON.parse(new TextDecoder().decode(pinned.indexBytes)).maxPieceBytes;
  const laneBytes = pinned.maxPartBytes + 2 * pieceBytes;
  const memoryBytes = concurrency * laneBytes;
  if (!Number.isSafeInteger(concurrency) || concurrency < 1
    || !Number.isSafeInteger(pinned.maxPartBytes) || pinned.maxPartBytes < 1
    || !Number.isSafeInteger(pieceBytes) || pieceBytes < 1
    || !Number.isSafeInteger(memoryBytes) || !Number.isSafeInteger(pinned.memoryBudgetBytes)
    || pinned.memoryBudgetBytes < memoryBytes) {
    throw new Error('Gather budget must cover every concurrent output and complete-piece buffer.');
  }
  const protectedCache = await prepareProtectedCache(options.protectedCache, pieceBytes);
  /** @type {Array<{ reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>, busy: boolean }>} */
  const lanes = [];
  try {
    for (let index = 0; index < concurrency; index++) {
      pinned.startupSignal?.throwIfAborted();
      lanes.push({ reader: await createVerifiedPieceGatherLane({ ...pinned, memoryBudgetBytes: laneBytes, protectedCache }), busy: false });
      pinned.startupSignal?.throwIfAborted();
    }
  } catch (error) {
    const outcomes = await Promise.allSettled(lanes.map(async lane => lane.reader.close()));
    const failures = outcomes.flatMap(outcome => outcome.status === 'rejected' ? [outcome.reason] : []);
    if (failures.length) throw new AggregateError([error, ...failures], 'Gather startup and cleanup failed.');
    throw error;
  }
  let closed = false, collecting = false;
  /** @type {Promise<void> | null} */
  let closing = null;
  /** @param {(reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>) => Promise<import('./verified-piece-gather.js').VerifiedRangeLease>} open
   * @param {AbortSignal | undefined} signal */
  const read = async (open, signal) => {
    signal?.throwIfAborted();
    if (closed) throw new Error('Gather reader closed.');
    const lane = collecting ? null : lanes.find(candidate => !candidate.busy);
    if (!lane) throw new Error('Release a current gather lease before another read.');
    lane.busy = true;
    try {
      const owned = await open(lane.reader);
      if (closed) { owned.release(); throw new Error('Gather reader closed.'); }
      let released = false;
      return Object.freeze({ bytes: owned.bytes, ...(owned.timings === undefined ? {} : { timings: owned.timings }), release() {
        if (released) return;
        released = true;
        try { owned.release(); } finally { lane.busy = false; }
      } });
    } catch (error) { lane.busy = false; throw error; }
  };
  /** @type {Pick<Awaited<ReturnType<import('./verified-piece-gather.js').createVerifiedPieceGather>>, 'readLease' | 'readManyLease' | 'getReceipt' | 'close'>} */
  const result = {
    readLease: (path, offset, length, signal) => read(reader => reader.readLease(path, offset, length, signal), signal),
    readManyLease: (ranges, signal) => read(reader => reader.readManyLease(ranges, signal), signal),
    async getReceipt() {
      if (closed) throw new Error('Gather reader closed.');
      if (collecting || lanes.some(lane => lane.busy)) throw new Error('Release all current gather leases before reading receipts.');
      collecting = true;
      try {
        // Await all owners even if one failed, before allowing another read.
        const outcomes = await Promise.allSettled(lanes.map(lane => lane.reader.getReceipt()));
        const failed = outcomes.find(outcome => outcome.status === 'rejected');
        if (failed) throw failed.reason;
        if (closed) throw new Error('Gather reader closed.');
        const receipts = outcomes.flatMap(outcome => outcome.status === 'fulfilled' ? [outcome.value] : []);
        const merged = { ...receipts[0] };
        const fields = /** @type {const} */ (['requestedBytes', 'acquisitionMs', 'verificationMs', 'copyMs', 'activeReadBytes', 'peakReadBytes', 'directPieceReads', 'directPieceBytes', 'batchReads', 'batchRanges', 'batchBytes', 'activeDigestCopyBytes', 'peakDigestCopyBytes', 'activeGenericScratchBytes', 'peakGenericScratchBytes', 'protectedCacheHits', 'protectedCacheHitBytes', 'protectedCacheMisses', 'protectedCacheAdmissions', 'protectedCacheAdmittedBytes', 'protectedCacheEvictions', 'protectedCacheEvictedBytes', 'protectedCacheSkips']);
        Object.assign(merged, Object.fromEntries(fields.map(field => [field, receipts.reduce((total, receipt) => total + receipt[field], 0)])));
        /** @type {Map<string, import('./verified-piece-reader.js').VerifiedPiece>} */
        const ranges = new Map(receipts.flatMap(receipt => receipt.ranges.map(range => [`${range.path}:${range.offset}`, range])));
        merged.ranges = [...ranges.values()];
        const pieces = new Map(merged.ranges.map(range => [range.identity, range.size]));
        merged.pieces = [...pieces.keys()];
        merged.verifiedBytes = [...pieces.values()].reduce((total, bytes) => total + bytes, 0);
        return merged;
      } finally { collecting = false; }
    },
    close() {
      if (!closing) {
        closed = true;
        const operation = (async () => {
          const outcomes = await Promise.allSettled(lanes.map(async lane => lane.reader.close()));
          const failed = outcomes.find(outcome => outcome.status === 'rejected');
          if (failed) throw failed.reason;
        })();
        closing = operation;
        operation.catch(() => { if (closing === operation) closing = null; });
      }
      return closing;
    },
  };
  const manifest = JSON.parse(new TextDecoder().decode(pinned.manifestBytes));
  /** @type {import('./verified-piece-gather.js').VerifiedRangeLeasePort} */
  const rangeLease = { concurrency, maxRanges, maxPartBytes: pinned.maxPartBytes, memoryBytes,
    fileIdentitiesBound: pinned.fileIdentities !== undefined,
    ...(protectedCache ? { protectedCacheBound: true } : {}),
    ...(pinned.fileIdentities !== undefined ? { batchFileIdentityChecks: /** @type {const} */ ('before-read-after-auth/v1') } : {}),
    read(shard, offset, length, signal) {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      return result.readLease(file.filename, offset, length, signal);
    },
    readMany(ranges, signal) {
      if (!Array.isArray(ranges) || ranges.length < 1 || ranges.length > maxRanges) return Promise.reject(new Error('Gather batch exceeds its admitted ranges bound.'));
      const selected = [];
      for (const range of ranges) {
        const file = Number.isSafeInteger(range?.shard) && range.shard >= 0 ? manifest.shards[range.shard] : null;
        if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
        selected.push({ path: file.filename, offset: range.offset, length: range.length });
      }
      return result.readManyLease(selected, signal);
    } };
  return Object.freeze({ ...result, rangeLease: Object.freeze(rangeLease) });
}
