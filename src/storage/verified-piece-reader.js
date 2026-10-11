/** @param {ArrayBuffer | Uint8Array<ArrayBuffer>} bytes */
const digest = async bytes => 'sha256:' + [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(byte => byte.toString(16).padStart(2, '0')).join('');

/** A host-pinned piece index authenticates ranges without acquiring their entire
 * storage shard. Acquisition/caching is a host port; every returned byte is verified.
 * This reader authenticates bytes; it does not establish signed Capsule qualification.
 * @type {import('./verified-piece-reader.js').createVerifiedPieceReader}
 */
export async function createVerifiedPieceReader({ manifestBytes, indexBytes, indexIdentity, acquire, signal }) {
  for (const bytes of [manifestBytes, indexBytes]) {
    if (!(bytes instanceof Uint8Array) && !(bytes instanceof ArrayBuffer)) throw new Error('Piece metadata requires bytes.');
  }
  manifestBytes = new Uint8Array(manifestBytes instanceof Uint8Array ? manifestBytes : new Uint8Array(manifestBytes)).buffer;
  indexBytes = new Uint8Array(indexBytes instanceof Uint8Array ? indexBytes : new Uint8Array(indexBytes)).buffer;
  if (await digest(indexBytes) !== indexIdentity) throw new Error('Piece index identity mismatch.');
  const index = JSON.parse(new TextDecoder().decode(indexBytes));
  if (index.schema !== 'doppler.verified-pieces/v1' || index.manifestIdentity !== await digest(manifestBytes)
    || !Array.isArray(index.files) || !Number.isSafeInteger(index.maxPieceBytes) || index.maxPieceBytes < 1) throw new Error('Piece index manifest binding mismatch.');
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  const files = new Map();
  for (const file of index.files) {
    if (typeof file.path !== 'string' || !/^[a-zA-Z0-9_.-]+$/.test(file.path) || files.has(file.path)
      || !Number.isSafeInteger(file.size) || file.size <= 0 || !Array.isArray(file.pieces)) throw new Error('Invalid piece file.');
    let offset = 0;
    for (const piece of file.pieces) {
      if (piece.offset !== offset || !Number.isSafeInteger(piece.size) || piece.size < 1
        || piece.size > index.maxPieceBytes || !/^sha256:[a-f0-9]{64}$/.test(piece.identity)) throw new Error('Invalid piece geometry.');
      offset += piece.size;
      if (!Number.isSafeInteger(offset) || offset > file.size) throw new Error('Piece coverage exceeds its file.');
    }
    if (offset !== file.size) throw new Error('Piece coverage mismatch.');
    files.set(file.path, file);
  }
  for (const shard of manifest.shards) {
    if (files.get(shard.filename)?.size !== shard.size) throw new Error('Piece index shard mismatch.');
  }
  const lifetime = new AbortController();
  signal = AbortSignal.any([lifetime.signal, ...(signal ? [signal] : [])]);
  const activeReads = new Set();
  let closed = false;
  /** @type {Promise<void> | null} */
  let closing = null;
  /** Settle the owned wrapper, without claiming an uncooperative host has stopped.
   * @template T
   * @param {Promise<T>} task
   * @returns {Promise<T>} */
  async function waitActive(task) {
    const current = /** @type {AbortSignal} */ (signal);
    /** @type {(() => void) | undefined} */
    let interrupted;
    try {
      return await Promise.race([task, new Promise((_, reject) => {
        interrupted = () => reject(current.reason);
        if (current.aborted) interrupted();
        else current.addEventListener('abort', interrupted, { once: true });
      })]);
    } finally { if (interrupted) current.removeEventListener('abort', interrupted); }
  }
  /** @type {import('./verified-piece-reader.js').VerifiedPieceReceipt} */
  const receipt = { indexIdentity, manifestIdentity: index.manifestIdentity, requestedBytes: 0, verifiedBytes: 0, pieces: [], ranges: [],
    acquisitionMs: 0, verificationMs: 0, copyMs: 0, activeReadBytes: 0, peakReadBytes: 0 };
  const seen = new Set(), ranges = new Set();
  /** @param {string} path @param {number} offset @param {number | null} length */
  async function readOwnedRange(path, offset, length) {
    signal?.throwIfAborted();
    const file = files.get(path);
    if (length === null) length = file ? file.size - offset : 0;
    if (closed || !file || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length)
      || length < 0 || !Number.isSafeInteger(offset + length) || offset + length > file.size) throw new Error('Piece range outside its declared file.');
    const output = new Uint8Array(length);
    receipt.activeReadBytes += length;
    receipt.peakReadBytes = Math.max(receipt.peakReadBytes, receipt.activeReadBytes);
    try {
      for (const piece of file.pieces) {
        if (piece.offset >= offset + length || piece.offset + piece.size <= offset) continue;
        signal?.throwIfAborted();
        const acquisitionStarted = performance.now();
        const received = await waitActive(acquire(Object.freeze({ ...piece, path }), { signal }));
        receipt.acquisitionMs += performance.now() - acquisitionStarted;
        const verificationStarted = performance.now();
        if (!(received instanceof Uint8Array) && !(received instanceof ArrayBuffer)
          || received.byteLength !== piece.size) throw new Error('Acquired piece integrity mismatch.');
        const bytes = new Uint8Array(received instanceof Uint8Array ? received : new Uint8Array(received));
        if (await waitActive(digest(bytes)) !== piece.identity) throw new Error('Acquired piece integrity mismatch.');
        receipt.verificationMs += performance.now() - verificationStarted;
        signal?.throwIfAborted();
        if (closed) throw new Error('Piece storage closed.');
        const start = Math.max(offset, piece.offset), end = Math.min(offset + length, piece.offset + piece.size);
        const copyStarted = performance.now();
        output.set(bytes.subarray(start - piece.offset, end - piece.offset), start - offset);
        receipt.copyMs += performance.now() - copyStarted;
        receipt.requestedBytes += end - start;
        const rangeKey = `${path}:${piece.offset}`;
        if (!ranges.has(rangeKey)) {
          ranges.add(rangeKey); receipt.ranges.push({ path, offset: piece.offset, size: piece.size, identity: piece.identity });
        }
        if (!seen.has(piece.identity)) { seen.add(piece.identity); receipt.verifiedBytes += bytes.byteLength; receipt.pieces.push(piece.identity); }
      }
      return output.buffer;
    } finally { receipt.activeReadBytes -= length; }
  }
  /** @param {string} path @param {number} offset @param {number | null} length */
  function readRange(path, offset, length) {
    const task = readOwnedRange(path, offset, length);
    activeReads.add(task);
    const settled = () => activeReads.delete(task);
    task.then(settled, settled);
    return task;
  }
  /** @param {string} path */
  const readBinary = path => {
    const file = files.get(path);
    if (!file) throw new Error('Undeclared auxiliary artifact: ' + path);
    return readRange(path, 0, file.size);
  };
  return Object.freeze({ manifest, readRange, readBinary,
    getReceipt: () => structuredClone(receipt),
    async preflight() {
      signal?.throwIfAborted();
      if (closed) throw new Error('Piece storage closed.');
    },
    close() {
      if (!closing) {
        closed = true;
        lifetime.abort(new Error('Piece storage closed.'));
        closing = (async () => {
          while (activeReads.size) await Promise.allSettled([...activeReads]);
        })();
      }
      return closing;
    },
  });
}
