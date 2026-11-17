import { createVerifiedPieceGatherLane } from './verified-piece-gather-lane.js';

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
    || !Number.isSafeInteger(memoryBytes) || !Number.isSafeInteger(pinned.memoryBudgetBytes)
    || pinned.memoryBudgetBytes < memoryBytes) {
    throw new Error('Gather budget must cover every concurrent output and complete-piece buffer.');
  }
  /** @type {Array<{ reader: Awaited<ReturnType<typeof createVerifiedPieceGatherLane>>, busy: boolean }>} */
  const lanes = [];
  try {
    for (let index = 0; index < concurrency; index++) {
      pinned.startupSignal?.throwIfAborted();
      lanes.push({ reader: await createVerifiedPieceGatherLane({ ...pinned, memoryBudgetBytes: laneBytes }), busy: false });
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
        const fields = /** @type {const} */ (['requestedBytes', 'acquisitionMs', 'verificationMs', 'copyMs', 'activeReadBytes', 'peakReadBytes', 'directPieceReads', 'directPieceBytes', 'batchReads', 'batchRanges', 'batchBytes', 'activeDigestCopyBytes', 'peakDigestCopyBytes', 'activeGenericScratchBytes', 'peakGenericScratchBytes']);
        for (const field of fields) {
          merged[field] = receipts.reduce((total, receipt) => total + receipt[field], 0);
        }
        /** @type {Map<string, import('./verified-piece-reader.js').VerifiedPiece>} */
        const ranges = new Map();
        for (const receipt of receipts) for (const range of receipt.ranges) ranges.set(`${range.path}:${range.offset}`, range);
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
