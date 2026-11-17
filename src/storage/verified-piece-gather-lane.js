/*
 * Node-only bounded whole-part gathering. The host explicitly pins the index and
 * maps authenticated filenames to approved local files. A lease remains owned by
 * the caller until release; release is safe after queue.writeBuffer returns.
 */
import { snapshotVerifiedPieceJobTimings, snapshotVerifiedRangeLeaseTimings } from './verified-piece-timings.js';
/** @type {import('./verified-piece-gather.js').createVerifiedPieceGather} */
export async function createVerifiedPieceGatherLane(options) {
  const { maxPartBytes, memoryBudgetBytes, indexIdentity } = options;
  const maxRanges = options.maxRanges === undefined ? 3 : options.maxRanges;
  if (!Number.isSafeInteger(maxRanges) || maxRanges < 1 || maxRanges > 32) {
    throw new Error('Gather maxRanges must be a positive safe integer no greater than 32.');
  }
  const manifestBytes = new Uint8Array(options.manifestBytes);
  const indexBytes = new Uint8Array(options.indexBytes);
  const files = { ...options.files };
  const index = JSON.parse(new TextDecoder().decode(indexBytes));
  const pieceBytes = index.maxPieceBytes;
  if (!Number.isSafeInteger(maxPartBytes) || maxPartBytes < 1
    || !Number.isSafeInteger(pieceBytes) || pieceBytes < 1
    || !Number.isSafeInteger(maxPartBytes + 2 * pieceBytes) || !Number.isSafeInteger(memoryBudgetBytes) || memoryBudgetBytes < maxPartBytes + 2 * pieceBytes) {
    throw new Error('Gather budget must hold one output and two complete piece buffers.');
  }
  // Acquire the output before starting a worker so allocation failure cannot
  // strand an unregistered thread during asynchronous factory admission.
  let buffer = new ArrayBuffer(maxPartBytes);
  // Metadata is copied by workerData; authentication completes before admission.
  const { Worker } = await import('node:worker_threads');
  const worker = new Worker(new URL('./verified-piece-gather-worker.js', import.meta.url), {
    workerData: { files, maxPartBytes, maxRanges, fileIdentities: options.fileIdentities, metadata: { manifestBytes, indexBytes, indexIdentity } },
  });
  /** A transferred/detached output is still worker-owned; only clear attached
   * buffers that have actually returned to this lane. @param {unknown} candidate */
  const clearOwned = candidate => {
    if (candidate instanceof ArrayBuffer && candidate.byteLength > 0) new Uint8Array(candidate).fill(0);
  };
  let closed = false;
  let workerExited = false, acknowledgedClose = false;
  let busy = false;
  /** @type {((error: Error) => void) | null} */
  let failPending = null;
  /** @type {((message: { ready?: boolean, buffer: ArrayBuffer, error?: string, timings?: unknown, receipt?: import('./verified-piece-gather.js').VerifiedPieceGatherReceipt }) => void) | null} */
  let receive = null;
  /** @type {Promise<void>} */
  const ready = new Promise((resolve, reject) => {
    failPending = reject;
    receive = message => { if (message.ready) { failPending = null; resolve(); } };
  });
  worker.on('message', message => {
    if (closed && message.buffer instanceof ArrayBuffer) { clearOwned(message.buffer); return; }
    receive?.(message);
  });
  /** @param {Error} error */
  const fail = error => {
    closed = true;
    // Pending transfers belong to the worker; published leases own their own
    // captured buffer until release. The closed pool retains neither allocation.
    buffer = new ArrayBuffer(0);
    failPending?.(error);
    failPending = null;
    receive = null;
  };
  worker.on('error', fail);
  worker.on('exit', () => { workerExited = true; fail(new Error('Gather worker closed.')); });
  try { await ready; } catch (error) { await worker.terminate(); throw error; }
  /** @type {Promise<void> | null} */
  let closing = null;
  const acknowledgeRetirement = () => new Promise((resolve, reject) => {
    /** @param {Error | null} error */
    const finish = error => {
      worker.off('message', onMessage); worker.off('error', onError); worker.off('exit', onExit);
      if (error) reject(error);
      else { acknowledgedClose = true; resolve(undefined); }
    };
    /** @param {{closed?: boolean, error?: string}} message */
    const onMessage = message => {
      if (message.closed === true) finish(message.error ? new Error(message.error) : null);
    };
    /** @param {Error} error */
    const onError = error => finish(error);
    const onExit = () => finish(new Error('Gather worker exited before acknowledging settled reads.'));
    worker.on('message', onMessage); worker.once('error', onError); worker.once('exit', onExit);
    if (workerExited) onExit();
    else {
      try { worker.postMessage({ close: true }); }
      catch (error) { finish(error); }
    }
  });
  /** @param {ReadonlyArray<{path: string, offset: number, length: number}>} ranges
   * @param {AbortSignal | undefined} signal @param {boolean} batch */
  const read = async (ranges, signal, batch) => {
      signal?.throwIfAborted();
      if (closed) throw new Error('Gather reader closed.');
      if (busy) throw new Error('Release the current gather lease before another read.');
      if (!Array.isArray(ranges) || ranges.length < 1 || ranges.length > maxRanges) {
        throw new Error('Gather batch exceeds its admitted ranges bound.');
      }
      const selected = ranges.map(range => ({ path: range.path, offset: range.offset, length: range.length }));
      const length = selected.reduce((sum, range) => sum + range.length, 0);
      if (selected.some(range => typeof range.path !== 'string' || !Number.isSafeInteger(range.offset) || range.offset < 0
          || !Number.isSafeInteger(range.length) || range.length < (batch ? 1 : 0)
          || !Number.isSafeInteger(range.offset + range.length))
          || !Number.isSafeInteger(length) || length > maxPartBytes) {
        throw new Error('Gather part exceeds its admitted output bound.');
      }
      busy = true;
      try {
        let roundTripMs = 0;
        /** @type {{ buffer: ArrayBuffer, error?: string, timings?: unknown }} */
        const message = await new Promise((resolve, reject) => {
          failPending = reject;
          let posted = 0;
          receive = result => { roundTripMs = performance.now() - posted; failPending = null; receive = null; resolve(result); };
          posted = performance.now();
          worker.postMessage({ ranges: selected, length, buffer, batch }, [buffer]);
        });
        if (closed) { clearOwned(message.buffer); throw new Error('Gather reader closed.'); }
        if (!(message.buffer instanceof ArrayBuffer) || message.buffer.byteLength !== maxPartBytes) throw new Error('Invalid gathered output ownership.');
        buffer = message.buffer;
        if (message.error) throw new Error(message.error);
        signal?.throwIfAborted();
        if (closed) throw new Error('Gather reader closed.');
        const workerTimings = snapshotVerifiedPieceJobTimings(message.timings);
        const timings = workerTimings === undefined ? undefined
          : snapshotVerifiedRangeLeaseTimings({ ...workerTimings, roundTripMs });
        const owned = buffer;
        let released = false;
        return Object.freeze({ bytes: new Uint8Array(owned, 0, length), ...(timings === undefined ? {} : { timings }), release() {
          if (released) return;
          released = true;
          // Detach the old views immediately so stale leases cannot mutate the pool.
          const recycled = structuredClone(owned, { transfer: [owned] });
          if (!closed) buffer = recycled;
          busy = false;
        } });
      } catch (error) {
        clearOwned(buffer);
        busy = false;
        throw error;
      }
    };
/** @type {Pick<Awaited<ReturnType<import('./verified-piece-gather.js').createVerifiedPieceGather>>, 'readLease' | 'readManyLease' | 'close' | 'getReceipt'>} */
  const result = {
    readLease: (path, offset, length, signal) => read([{ path, offset, length }], signal, false),
    readManyLease: (ranges, signal) => read(ranges, signal, true),
    async getReceipt() {
      if (closed) throw new Error('Gather reader closed.');
      if (busy) throw new Error('Release the current gather lease before reading its receipt.');
      busy = true;
      try {
        return await new Promise((resolve, reject) => {
          failPending = reject;
          receive = message => {
            failPending = null; receive = null;
            if (!message.receipt) reject(new Error('Missing gather receipt.'));
            else resolve(message.receipt);
          };
          worker.postMessage({ receipt: true });
        });
      } finally { busy = false; }
    },
    close() {
      if (!closing) {
        fail(new Error('Gather reader closed.'));
        const operation = (async () => {
          if (!acknowledgedClose) await acknowledgeRetirement();
          await worker.terminate();
        })();
        closing = operation;
        operation.catch(() => { if (closing === operation) closing = null; });
      }
      return closing;
    },
  };
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
/** @type {import('./verified-piece-gather.js').VerifiedRangeLeasePort} */
  const rangeLease = { maxPartBytes, maxRanges, memoryBytes: maxPartBytes + 2 * pieceBytes,
    read: (shard, offset, length, signal) => {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      return result.readLease(file.filename, offset, length, signal);
    },
  };
  return Object.freeze({ ...result, rangeLease: Object.freeze(rangeLease) });
}
