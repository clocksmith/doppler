import { parentPort, workerData } from 'node:worker_threads';
import { openSync, readSync, closeSync, fstatSync, lstatSync, constants } from 'node:fs';
import { createHash } from 'node:crypto';
import { createVerifiedPieceReader } from './verified-piece-reader.js';

// This worker owns acquisition, complete-piece authentication and range gathering.
// Only the finished range crosses the thread boundary. No caller scratch is detached.
/** @type {Map<string, number>} */
const handles = new Map();
if (!parentPort) throw new Error('Gather requires a worker port.');
const port = parentPort;
/** @param {string} path @param {number} handle */
function assertIdentity(path, handle) {
  const expected = workerData.fileIdentities?.[path];
  if (!expected) return;
  /** @param {import('node:fs').BigIntStats} stat */
  const matches = stat => stat.isFile() && stat.dev.toString() === expected.device
    && stat.ino.toString() === expected.inode && stat.size.toString() === expected.size;
  if (!matches(fstatSync(handle, { bigint: true }))
    || !matches(lstatSync(workerData.files[path], { bigint: true }))) {
    throw new Error('Gather source identity changed after local approval.');
  }
}
/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} bytes */
function readFile(piece, bytes) {
  if (bytes.byteLength !== piece.size) throw new Error('Direct source read must cover its complete piece.');
  const filename = workerData.files[piece.path];
  if (typeof filename !== 'string') throw new Error('Missing approved piece file.');
  let handle = handles.get(piece.path);
  if (handle === undefined) {
    if (!Number.isInteger(constants.O_NOFOLLOW) || !Number.isInteger(constants.O_NONBLOCK)) {
      throw new Error('Gather regular-file admission is unsupported on this platform.');
    }
    // A replaced leaf must not follow a symlink or block admission on a FIFO.
    // Once admitted, the retained FD owns this regular file; every read still
    // authenticates complete piece bytes against the pinned index.
    handle = openSync(filename, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    try {
      if (!fstatSync(handle).isFile()) throw new Error('Gather source must be a regular file.');
      assertIdentity(piece.path, handle);
    } catch (error) {
      try { closeSync(handle); }
      catch (cleanup) { throw new AggregateError([error, cleanup], 'Gather file admission and cleanup failed.'); }
      throw error;
    }
    handles.set(piece.path, handle);
  } else assertIdentity(piece.path, handle);
  let read = 0;
  while (read < bytes.length) {
    const count = readSync(handle, bytes, read, bytes.length - read, piece.offset + read);
    if (!count) throw new Error('Incomplete authenticated piece.');
    read += count;
  }
  assertIdentity(piece.path, handle);
}
const reader = await createVerifiedPieceReader({ ...workerData.metadata,
  acquire: async piece => {
    const bytes = new Uint8Array(piece.size);
    readFile(piece, bytes);
    return bytes;
  },
});
// Only the common verifier's authenticated index may select the direct path.
// This worker owns the FD and transferred output; no external acquisition callback
// can mutate the bytes between synchronous full-piece SHA and publication.
let directPieceReads = 0, directPieceBytes = 0, acquisitionMs = 0, verificationMs = 0;
/** @type {Map<string, Readonly<import('./verified-piece-reader.js').VerifiedPiece>>} */
const directRanges = new Map();
/** @returns {import('./verified-piece-gather.js').VerifiedPieceGatherReceipt} */
function getReceipt() {
  const generic = reader.getReceipt();
  const ranges = new Map(generic.ranges.map(range => [`${range.path}:${range.offset}`, range]));
  for (const [key, range] of directRanges) ranges.set(key, range);
  const pieces = new Map([...ranges.values()].map(range => [range.identity, range.size]));
  return { ...generic, ranges: [...ranges.values()], pieces: [...pieces.keys()],
    verifiedBytes: [...pieces.values()].reduce((sum, size) => sum + size, 0),
    requestedBytes: generic.requestedBytes + directPieceBytes,
    acquisitionMs: generic.acquisitionMs + acquisitionMs, verificationMs: generic.verificationMs + verificationMs,
    directPieceReads, directPieceBytes };
}
port.postMessage({ ready: true });
port.on('message', async ({ path, offset, length, buffer, receipt }) => {
  if (receipt === true) { port.postMessage({ receipt: getReceipt() }); return; }
  try {
    const output = new Uint8Array(buffer, 0, length);
    const piece = reader.resolveCompletePiece(path, offset, length);
    if (piece) {
      const readStart = performance.now();
      try { readFile(piece, output); }
      finally { acquisitionMs += performance.now() - readStart; }
      const hashStart = performance.now();
      const identity = 'sha256:' + createHash('sha256').update(output).digest('hex');
      verificationMs += performance.now() - hashStart;
      if (identity !== piece.identity) throw new Error('Acquired piece integrity mismatch.');
      directRanges.set(`${path}:${offset}`, piece);
      directPieceReads++; directPieceBytes += length;
    } else await reader.readInto(path, offset, output);
    port.postMessage({ buffer, length }, [buffer]);
  } catch (error) {
    // Never publish a verified prefix after a later piece fails verification.
    new Uint8Array(buffer).fill(0);
    port.postMessage({ buffer, error: String(error.message ?? error) }, [buffer]);
  }
});
process.on('exit', () => { for (const handle of handles.values()) closeSync(handle); });
