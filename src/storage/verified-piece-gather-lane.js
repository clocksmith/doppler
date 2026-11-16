/*
 * Node-only bounded whole-part gathering. The host explicitly pins the index and
 * maps authenticated filenames to approved local files. A lease remains owned by
 * the caller until release; release is safe after queue.writeBuffer returns.
 */
/** @type {import('./verified-piece-gather.js').createVerifiedPieceGather} */
export async function createVerifiedPieceGatherLane(options) {
  const { maxPartBytes, memoryBudgetBytes, indexIdentity } = options;
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
    workerData: { files, fileIdentities: options.fileIdentities, metadata: { manifestBytes, indexBytes, indexIdentity } },
  });
  let closed = false;
  let busy = false;
  /** @type {((error: Error) => void) | null} */
  let failPending = null;
  /** @type {((message: { ready?: boolean, buffer: ArrayBuffer, error?: string, receipt?: import('./verified-piece-gather.js').VerifiedPieceGatherReceipt }) => void) | null} */
  let receive = null;
  /** @type {Promise<void>} */
  const ready = new Promise((resolve, reject) => {
    failPending = reject;
    receive = message => { if (message.ready) { failPending = null; resolve(); } };
  });
  worker.on('message', message => receive?.(message));
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
  worker.on('exit', () => fail(new Error('Gather worker closed.')));
  try { await ready; } catch (error) { await worker.terminate(); throw error; }
  /** @type {Promise<void> | null} */
  let closing = null;
/** @type {Pick<Awaited<ReturnType<import('./verified-piece-gather.js').createVerifiedPieceGather>>, 'readLease' | 'close' | 'getReceipt'>} */
  const result = {
    async readLease(path, offset, length, signal) {
      signal?.throwIfAborted();
      if (closed) throw new Error('Gather reader closed.');
      if (busy) throw new Error('Release the current gather lease before another read.');
      if (!Number.isSafeInteger(length) || length < 0 || length > maxPartBytes) {
        throw new Error('Gather part exceeds its admitted output bound.');
      }
      busy = true;
      try {
        /** @type {{ buffer: ArrayBuffer, error?: string }} */
        const message = await new Promise((resolve, reject) => {
          failPending = reject;
          receive = result => { failPending = null; receive = null; resolve(result); };
          worker.postMessage({ path, offset, length, buffer }, [buffer]);
        });
        if (closed) throw new Error('Gather reader closed.');
        buffer = message.buffer;
        if (message.error) throw new Error(message.error);
        signal?.throwIfAborted();
        if (closed) throw new Error('Gather reader closed.');
        const owned = buffer;
        let released = false;
        return Object.freeze({ bytes: new Uint8Array(owned, 0, length), release() {
          if (released) return;
          released = true;
          // Detach the old views immediately so stale leases cannot mutate the pool.
          const recycled = structuredClone(owned, { transfer: [owned] });
          if (!closed) buffer = recycled;
          busy = false;
        } });
      } catch (error) {
        busy = false;
        throw error;
      }
    },
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
        const operation = worker.terminate().then(() => {});
        closing = operation;
        operation.catch(() => { if (closing === operation) closing = null; });
      }
      return closing;
    },
  };
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
/** @type {import('./verified-piece-gather.js').VerifiedRangeLeasePort} */
  const rangeLease = { maxPartBytes, memoryBytes: maxPartBytes + 2 * pieceBytes,
    read: (shard, offset, length, signal) => {
      const file = manifest.shards[shard];
      if (!file) return Promise.reject(new Error('Undeclared gather shard.'));
      return result.readLease(file.filename, offset, length, signal);
    },
  };
  return Object.freeze({ ...result, rangeLease: Object.freeze(rangeLease) });
}
