import { createVerifiedPieceGatherLane } from './verified-piece-gather-lane.js';
import { createVerifiedDirectReader } from './verified-piece-direct.js';
import { prepareVerifiedPieceStateCache } from './verified-piece-state-cache.js';

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
  const index = JSON.parse(new TextDecoder().decode(pinned.indexBytes));
  const pieceBytes = index.maxPieceBytes;
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
  const ownedStateCache = options.sharedStateCache === undefined
    ? await prepareVerifiedPieceStateCache({
      rawPolicy: options.verificationCache,
      manifestIdentity: index.manifestIdentity,
      indexIdentity: pinned.indexIdentity,
      index,
      files: pinned.files,
    })
    : undefined;
  const stateCache = options.sharedStateCache ?? ownedStateCache;
  /** @type {Array<{ reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>, busy: boolean }>} */
  const lanes = [];
  try {
    pinned.startupSignal?.throwIfAborted();
    // Every lane authenticates the same pinned index in its own worker. Start them
    // together: admitting them one after another made startup cost one index
    // authentication per lane (about 0.75 s each for a 63 MB index).
    const started = await Promise.allSettled(Array.from({ length: concurrency }, (_, index) =>
      createVerifiedPieceGatherLane({
        ...pinned,
        memoryBudgetBytes: laneBytes,
        indexMaxPieceBytes: pieceBytes,
        protectedCache,
        stateCache: stateCache?.workerPayload,
        workerIndex: index,
        workerCount: concurrency,
        onStateDirty: stateCache ? () => stateCache.scheduleFlush() : undefined,
      })));
    // Register every lane that did start before reporting a failure, so cleanup owns them all.
    for (const outcome of started) {
      if (outcome.status === 'fulfilled') lanes.push({ reader: outcome.value, busy: false });
    }
    const failed = started.find(outcome => outcome.status === 'rejected');
    if (failed) throw failed.reason;
    pinned.startupSignal?.throwIfAborted();
  } catch (error) {
    const outcomes = await Promise.allSettled([
      ...lanes.map(async lane => lane.reader.close()),
      ...(ownedStateCache ? [ownedStateCache.close()] : []),
    ]);
    const failures = outcomes.flatMap(outcome => outcome.status === 'rejected' ? [outcome.reason] : []);
    if (failures.length) throw new AggregateError([error, ...failures], 'Gather startup and cleanup failed.');
    throw error;
  }
  // Ranges whose pieces are already verified for the current file state need no lane: the caller may
  // read them straight into its own memory. Bound file identities and a verification cache are both
  // required, because those are what the direct checks compare against.
  const direct = stateCache && pinned.fileIdentities !== undefined
    ? createVerifiedDirectReader({ files: pinned.files, fileIdentities: pinned.fileIdentities, index, stateCache }) : null;
  let closed = false, collecting = false;
  /** @type {Promise<void> | null} */
  let closing = null;
  // Background verification waits for a lane instead of failing. A foreground read keeps its contract
  // (more concurrent reads than lanes is the caller's error) and waits only when a lane is held by a
  // background job, which it is served ahead of.
  /** @typedef {{ reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>, busy: boolean, background?: boolean }} Lane */
  /** @type {Array<{ resolve: (lane: Lane) => void, reject: (error: unknown) => void, background: boolean }>} */
  const waiters = [];
  /** @param {Lane} lane */
  const releaseLane = lane => {
    let next = collecting ? -1 : waiters.findIndex(waiter => !waiter.background);
    if (next < 0 && !collecting && waiters.length) next = 0;
    if (next < 0) { lane.busy = false; lane.background = false; return; }
    const [waiter] = waiters.splice(next, 1);
    lane.background = waiter.background;
    waiter.resolve(lane);
  };
  /** @param {AbortSignal | undefined} signal @param {boolean} background @returns {Lane | Promise<Lane>} */
  const acquireLane = (signal, background) => {
    /** @type {Lane | undefined} */
    const lane = collecting ? undefined : lanes.find(candidate => !candidate.busy);
    if (lane) { lane.busy = true; lane.background = background; return lane; }
    if (!background && !lanes.some(/** @param {Lane} candidate */ candidate => candidate.background === true)) {
      throw new Error('Release a current gather lease before another read.');
    }
    return new Promise((resolve, reject) => {
      /** @param {Lane} acquired */
      const settle = acquired => { signal?.removeEventListener('abort', abort); resolve(acquired); };
      const waiter = { background, reject, resolve: settle };
      const abort = () => {
        const at = waiters.indexOf(waiter);
        if (at >= 0) { waiters.splice(at, 1); reject(signal?.reason); }
      };
      signal?.addEventListener('abort', abort, { once: true });
      waiters.push(waiter);
    });
  };
  /** @param {(reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>) => Promise<import('./verified-piece-gather.js').VerifiedRangeLease>} open
   * @param {AbortSignal | undefined} signal @param {boolean} [background] */
  const read = async (open, signal, background = false) => {
    signal?.throwIfAborted();
    if (closed) throw new Error('Gather reader closed.');
    // A free lane is taken without yielding, so the job is posted in the caller's own turn as before.
    const acquired = acquireLane(signal, background);
    const lane = acquired instanceof Promise ? await acquired : acquired;
    try {
      if (closed) throw new Error('Gather reader closed.');
      const owned = await open(lane.reader);
      if (closed) { owned.release(); throw new Error('Gather reader closed.'); }
      let released = false;
      return Object.freeze({ bytes: owned.bytes, ...(owned.timings === undefined ? {} : { timings: owned.timings }), release() {
        if (released) return;
        released = true;
        try { owned.release(); } finally { releaseLane(lane); }
      } });
    } catch (error) { releaseLane(lane); throw error; }
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
        const fields = /** @type {const} */ (['requestedBytes', 'acquisitionMs', 'verificationMs', 'copyMs', 'activeReadBytes', 'peakReadBytes', 'directPieceReads', 'directPieceBytes', 'batchReads', 'batchRanges', 'batchBytes', 'activeDigestCopyBytes', 'peakDigestCopyBytes', 'activeGenericScratchBytes', 'peakGenericScratchBytes', 'protectedCacheHits', 'protectedCacheHitBytes', 'protectedCacheMisses', 'protectedCacheAdmissions', 'protectedCacheAdmittedBytes', 'protectedCacheEvictions', 'protectedCacheEvictedBytes', 'protectedCacheSkips', 'stateCacheHits', 'stateCacheHitBytes', 'stateCacheMisses', 'stateCacheAdmissions', 'stateCacheAdmittedBytes', 'backgroundVerifiedPieces', 'backgroundVerificationMs']);
        Object.assign(merged, Object.fromEntries(fields.map(field => [field, receipts.reduce((total, receipt) => total + (receipt[field] ?? 0), 0)])));
        /** @type {Map<string, import('./verified-piece-reader.js').VerifiedPiece>} */
        const ranges = new Map(receipts.flatMap(receipt => receipt.ranges.map(range => [`${range.path}:${range.offset}`, range])));
        merged.ranges = [...ranges.values()];
        const pieces = new Map(merged.ranges.map(range => [range.identity, range.size]));
        merged.pieces = [...pieces.keys()];
        merged.verifiedBytes = [...pieces.values()].reduce((total, bytes) => total + bytes, 0);
        if (direct) {
          // Bytes served from already verified pieces without a lane; not part of ranges or verifiedBytes.
          const served = direct.getStats();
          Object.assign(merged, { directVerifiedReads: served.reads, directVerifiedBytes: served.bytes, directDeclinedReads: served.declined,
            directAheadReads: served.aheadReads, directAheadBytes: served.aheadBytes, directAheadConfirmed: served.aheadConfirmed });
        }
        return merged;
      } finally {
        collecting = false;
        // Background jobs that arrived meanwhile take the lanes that are free now.
        for (const lane of lanes) if (!lane.busy && waiters.length) { lane.busy = true; releaseLane(lane); }
      }
    },
    close() {
      if (!closing) {
        closed = true;
        for (const waiter of waiters.splice(0)) waiter.reject(new Error('Gather reader closed.'));
        const operation = (async () => {
          const outcomes = await Promise.allSettled([...lanes.map(async lane => lane.reader.close()),
            ...(direct ? [direct.close()] : [])]);
          if (ownedStateCache) await ownedStateCache.close();
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
    ...(stateCache ? { verificationCacheBound: true } : {}),
    ...(pinned.fileIdentities !== undefined ? { batchFileIdentityChecks: /** @type {const} */ ('before-read-after-auth/v1') } : {}),
    read(shard, offset, length, signal) {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      return result.readLease(file.filename, offset, length, signal);
    },
    ...(direct ? { readVerifiedInto(shard, offset, destination, signal) {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      if (closed) return Promise.reject(new Error('Gather reader closed.'));
      return direct.readInto(file.filename, offset, destination, signal);
    }, readAheadInto(shard, offset, destination, signal) {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      if (closed) return Promise.reject(new Error('Gather reader closed.'));
      return direct.readAheadInto(file.filename, offset, destination, signal);
    }, hasVerified(shard, offset, length) {
      const file = manifest.shards[shard];
      return !!file && !closed && direct.isVerified(file.filename, offset, length);
    } } : {}),
    readMany(ranges, signal) {
      if (!Array.isArray(ranges) || ranges.length < 1 || ranges.length > maxRanges) return Promise.reject(new Error('Gather batch exceeds its admitted ranges bound.'));
      const selected = [];
      for (const range of ranges) {
        const file = Number.isSafeInteger(range?.shard) && range.shard >= 0 ? manifest.shards[range.shard] : null;
        if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
        selected.push({ path: file.filename, offset: range.offset, length: range.length });
      }
      return result.readManyLease(selected, signal);
    },
    async verifyMany(ranges, signal) {
      if (!Array.isArray(ranges) || ranges.length < 1 || ranges.length > maxRanges) throw new Error('Gather batch exceeds its admitted ranges bound.');
      /** @type {Array<{ path: string, offset: number, length: number }>} */
      const selected = [];
      for (const range of ranges) {
        const file = Number.isSafeInteger(range?.shard) && range.shard >= 0 ? manifest.shards[range.shard] : null;
        if (!file) throw new Error('Undeclared gather shard.');
        selected.push({ path: file.filename, offset: range.offset, length: range.length });
      }
      const lease = await read(reader => reader.readManyLease(selected, signal), signal, true);
      const timings = lease.timings ?? null;
      lease.release();
      return timings;
    } };
  return Object.freeze({ ...result, rangeLease: Object.freeze(rangeLease) });
}
