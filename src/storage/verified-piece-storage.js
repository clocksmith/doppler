import { createArtifactStorageContext } from './artifact-storage-context.js';

/** @param {ArrayBuffer | Uint8Array<ArrayBuffer>} bytes */
const digest = async bytes => 'sha256:' + [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(byte => byte.toString(16).padStart(2, '0')).join('');

/** A host-pinned piece index authenticates ranges without acquiring their entire
 * storage shard. Acquisition/caching is a host port; every returned byte is verified.
 * This is an RDRR storage context, not signed Capsule qualification.
 * @type {import('./verified-piece-storage.js').createVerifiedPieceStorage}
 */
export async function createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity, acquire, signal }) {
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
    }
    if (offset !== file.size) throw new Error('Piece coverage mismatch.');
    files.set(file.path, file);
  }
  for (const shard of manifest.shards) {
    if (files.get(shard.filename)?.size !== shard.size) throw new Error('Piece index shard mismatch.');
  }
  let closed = false;
  /** @type {import('./verified-piece-storage.js').VerifiedPieceReceipt} */
  const receipt = { indexIdentity, manifestIdentity: index.manifestIdentity, requestedBytes: 0, verifiedBytes: 0, pieces: [],
    acquisitionMs: 0, verificationMs: 0, copyMs: 0, activeReadBytes: 0, peakReadBytes: 0 };
  const seen = new Set();
  /** @param {string} path @param {number} offset @param {number | null} length */
  async function readRange(path, offset, length) {
    signal?.throwIfAborted();
    const file = files.get(path);
    if (length === null) length = file ? file.size - offset : 0;
    if (closed || !file || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length)
      || length < 0 || offset + length > file.size) throw new Error('Piece range outside its declared file.');
    const output = new Uint8Array(length);
    receipt.activeReadBytes += length;
    receipt.peakReadBytes = Math.max(receipt.peakReadBytes, receipt.activeReadBytes);
    try {
      for (const piece of file.pieces) {
        if (piece.offset >= offset + length || piece.offset + piece.size <= offset) continue;
        signal?.throwIfAborted();
        const acquisitionStarted = performance.now();
        const received = await acquire(Object.freeze({ ...piece, path }), { signal });
        receipt.acquisitionMs += performance.now() - acquisitionStarted;
        const verificationStarted = performance.now();
        const bytes = received instanceof Uint8Array ? new Uint8Array(received) : new Uint8Array(received);
        if (bytes.byteLength !== piece.size || await digest(bytes) !== piece.identity) throw new Error('Acquired piece integrity mismatch.');
        receipt.verificationMs += performance.now() - verificationStarted;
        signal?.throwIfAborted();
        if (closed) throw new Error('Piece storage closed.');
        const start = Math.max(offset, piece.offset), end = Math.min(offset + length, piece.offset + piece.size);
        const copyStarted = performance.now();
        output.set(bytes.subarray(start - piece.offset, end - piece.offset), start - offset);
        receipt.copyMs += performance.now() - copyStarted;
        receipt.requestedBytes += end - start;
        if (!seen.has(piece.identity)) { seen.add(piece.identity); receipt.verifiedBytes += bytes.byteLength; receipt.pieces.push(piece.identity); }
      }
      return output.buffer;
    } finally { receipt.activeReadBytes -= length; }
  }
  /** @param {string} path */
  const readBinary = path => {
    const file = files.get(path);
    if (!file) throw new Error('Undeclared auxiliary artifact: ' + path);
    return readRange(path, 0, file.size);
  };
  const storage = createArtifactStorageContext({ manifest, readRange, readBinary,
    readText: async path => new TextDecoder().decode(await readBinary(path)),
    // The range reader verifies its independently pinned pieces before returning.
    // Whole-shard verification would acquire unrelated partition dependencies.
    verifyHashes: true, hashesTrusted: true, close: async () => { closed = true; } });
  // The authenticated index already checks file coverage. Probing every shard's
  // tail would acquire unrelated partitions before selective loading starts.
  storage.preflight = async () => {
    signal?.throwIfAborted();
    if (closed) throw new Error('Piece storage closed.');
  };
  return { manifest, storage, getReceipt: () => structuredClone(receipt) };
}
