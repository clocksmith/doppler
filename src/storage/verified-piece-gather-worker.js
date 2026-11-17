import { parentPort, workerData } from 'node:worker_threads';
import { openSync, readSync, read as readAsync, closeSync, fstatSync, lstatSync, constants } from 'node:fs';
import { createHash } from 'node:crypto';
import { createVerifiedPieceReader } from './verified-piece-reader.js';
import { snapshotVerifiedPieceJobTimings } from './verified-piece-timings.js';

// This worker owns acquisition, complete-piece authentication and range gathering.
// Only the finished range crosses the thread boundary. No caller scratch is detached.
/** @type {Map<string, number>} */
const handles = new Map();
if (!parentPort) throw new Error('Gather requires a worker port.');
const port = parentPort;
const maxRanges = workerData.maxRanges === undefined ? 3 : workerData.maxRanges;
if (!Number.isSafeInteger(maxRanges) || maxRanges < 1 || maxRanges > 32) {
  throw new Error('Gather maxRanges must be a positive safe integer no greater than 32.');
}
let closing = false;
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
function admitFile(piece, bytes) {
  if (closing) throw new Error('Gather worker closed.');
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
  return handle;
}
/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} bytes */
function readFile(piece, bytes) {
  const handle = admitFile(piece, bytes);
  let read = 0;
  while (read < bytes.length) {
    const count = readSync(handle, bytes, read, bytes.length - read, piece.offset + read);
    if (!count) throw new Error('Incomplete authenticated piece.');
    read += count;
  }
  assertIdentity(piece.path, handle);
}
/** Whole-piece batch outputs are disjoint owned views, so parallel reads need no
 * extra complete-piece scratch. Partial/generic acquisitions remain serialized.
 * @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} bytes @param {number | null} admittedHandle */
async function readFileParallel(piece, bytes, admittedHandle = null) {
  const handle = admittedHandle ?? admitFile(piece, bytes);
  let read = 0;
  while (read < bytes.length) {
    if (closing) throw new Error('Gather worker closed.');
    const count = await new Promise((resolve, reject) => {
      readAsync(handle, bytes, read, bytes.length - read, piece.offset + read,
        (error, bytesRead) => error ? reject(error) : resolve(bytesRead));
    });
    if (!count) throw new Error('Incomplete authenticated piece.');
    read += count;
  }
  if (admittedHandle === null) assertIdentity(piece.path, handle);
}
const reader = await createVerifiedPieceReader({ ...workerData.metadata,
  acquire: async () => { throw new Error('Gather worker uses owned piece verification.'); },
});
// Only the common verifier's authenticated index may select the direct path.
// This worker owns the FD and transferred output; no external acquisition callback
// can mutate the bytes between full-piece SHA and publication.
let directPieceReads = 0, directPieceBytes = 0, acquisitionMs = 0, verificationMs = 0;
let batchReads = 0, batchRanges = 0, batchBytes = 0;
let activeGenericScratchBytes = 0, peakGenericScratchBytes = 0, genericRequestedBytes = 0, genericCopyMs = 0;
/** @type {Map<string, Readonly<import('./verified-piece-reader.js').VerifiedPiece>>} */
const verifiedRanges = new Map();
const maxPieceBytes = JSON.parse(new TextDecoder().decode(workerData.metadata.indexBytes)).maxPieceBytes;
/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {string} identity */
function recordPiece(piece, identity) {
  if (identity !== piece.identity) throw new Error('Acquired piece integrity mismatch.');
  verifiedRanges.set(`${piece.path}:${piece.offset}`, piece);
}
/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {string} identity */
function recordDirect(piece, identity) {
  recordPiece(piece, identity);
  directPieceReads++; directPieceBytes += piece.size;
}
/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} output */
function authenticateDirect(piece, output) {
  const started = performance.now();
  const identity = 'sha256:' + createHash('sha256').update(output).digest('hex');
  verificationMs += performance.now() - started;
  recordDirect(piece, identity);
}
/** @returns {import('./verified-piece-gather.js').VerifiedPieceGatherReceipt} */
function getReceipt() {
  const generic = reader.getReceipt();
  const ranges = new Map(generic.ranges.map(range => [`${range.path}:${range.offset}`, range]));
  for (const [key, range] of verifiedRanges) ranges.set(key, range);
  const pieces = new Map([...ranges.values()].map(range => [range.identity, range.size]));
  return { ...generic, ranges: [...ranges.values()], pieces: [...pieces.keys()],
    verifiedBytes: [...pieces.values()].reduce((sum, size) => sum + size, 0),
    requestedBytes: generic.requestedBytes + directPieceBytes + genericRequestedBytes,
    copyMs: generic.copyMs + genericCopyMs,
    acquisitionMs: generic.acquisitionMs + acquisitionMs, verificationMs: generic.verificationMs + verificationMs,
    directPieceReads, directPieceBytes, batchReads, batchRanges, batchBytes,
    activeDigestCopyBytes: 0, peakDigestCopyBytes: 0,
    activeGenericScratchBytes, peakGenericScratchBytes };
}
port.postMessage({ ready: true });
/** @type {Set<Promise<void>>} */
const operations = new Set();
/** @type {Promise<void> | null} */
let retirement = null;
function closeWorker() {
  closing = true;
  if (!retirement) {
    const operation = (async () => {
      await Promise.allSettled([...operations]);
      await reader.close();
      const errors = [];
      for (const [path, handle] of handles) {
        try { closeSync(handle); handles.delete(path); }
        catch (error) { errors.push(error); }
      }
      if (errors.length) throw new AggregateError(errors, 'Gather descriptor retirement failed.');
    })();
    retirement = operation;
    operation.catch(() => { if (retirement === operation) retirement = null; });
  }
  return retirement;
}
/** @param {{ ranges: ReadonlyArray<{path: string, offset: number, length: number}>, length: number,
 * buffer: ArrayBuffer, batch?: boolean }} request */
async function gather({ ranges, length, buffer, batch }) {
  const started = performance.now();
  const before = { acquisitionMs, verificationMs, genericCopyMs };
  try {
    if (closing) throw new Error('Gather worker closed.');
    if (operations.size) throw new Error('Gather worker already owns a request.');
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== workerData.maxPartBytes
      || !Array.isArray(ranges) || ranges.length < 1 || ranges.length > maxRanges
      || !Number.isSafeInteger(length) || length < 0 || length > buffer.byteLength
      || ranges.some(range => typeof range.path !== 'string' || !Number.isSafeInteger(range.offset) || range.offset < 0
        || !Number.isSafeInteger(range.length) || range.length < (batch ? 1 : 0)
        || !Number.isSafeInteger(range.offset + range.length))
      || ranges.reduce((sum, range) => sum + range.length, 0) !== length) {
      throw new Error('Invalid bounded gather request.');
    }
    const pieces = ranges.map(({ path, offset, length: rangeLength }) => reader.resolveCompletePiece(path, offset, rangeLength));
    if (batch && pieces.every(piece => piece !== null)) {
      let destination = 0;
      const outputs = pieces.map(piece => {
        const output = new Uint8Array(buffer, destination, piece.size);
        destination += piece.size;
        return output;
      });
      // One job owns these exact descriptor bindings. Admission checks each
      // distinct path before any read; publication checks them again only after
      // every native read and digest settles. Never reuse guards across jobs.
      /** @type {Map<string, number>} */
      const admitted = new Map();
      for (const [index, piece] of pieces.entries()) {
        if (!admitted.has(piece.path)) admitted.set(piece.path, admitFile(piece, outputs[index]));
      }
      const reads = pieces.map((piece, index) => {
        const output = outputs[index];
        return (async () => {
          const readStart = performance.now();
          try { await readFileParallel(piece, output, admitted.get(piece.path)); }
          finally { acquisitionMs += performance.now() - readStart; }
          if (closing) throw new Error('Gather worker closed.');
          // This worker exclusively owns the output until transfer. Hash that
          // storage directly, without WebCrypto's input snapshot and another
          // shared thread-pool handoff. Other worker lanes still run in parallel.
          authenticateDirect(piece, output);
        })();
      });
      // A rejected read must not zero or transfer a buffer while a sibling native
      // read can still write it. Settle every admitted read before either outcome.
      const outcomes = await Promise.allSettled(reads);
      const failed = outcomes.find(outcome => outcome.status === 'rejected');
      const identityFailures = [];
      for (const [path, handle] of admitted) {
        try { assertIdentity(path, handle); } catch (error) { identityFailures.push(error); }
      }
      if (failed) throw failed.reason;
      if (identityFailures.length) throw identityFailures[0];
    } else {
      let destination = 0;
      /** @type {Uint8Array<ArrayBuffer> | null} */
      let scratch = null;
      try {
        for (const [index, { path, offset, length: rangeLength }] of ranges.entries()) {
          const output = new Uint8Array(buffer, destination, rangeLength);
          const piece = pieces[index];
          if (piece) {
            const readStart = performance.now();
            try { readFile(piece, output); }
            finally { acquisitionMs += performance.now() - readStart; }
            authenticateDirect(piece, output);
          } else {
            const selected = reader.resolveRangePieces(path, offset, rangeLength);
            if (!selected.length) { destination += rangeLength; continue; }
            if (!scratch) {
              scratch = new Uint8Array(maxPieceBytes);
              activeGenericScratchBytes = scratch.byteLength;
              peakGenericScratchBytes = Math.max(peakGenericScratchBytes, activeGenericScratchBytes);
            }
            for (const current of selected) {
              const bytes = scratch.subarray(0, current.size), readStart = performance.now();
              try { readFile(current, bytes); }
              finally { acquisitionMs += performance.now() - readStart; }
              const hashStart = performance.now();
              const identity = 'sha256:' + createHash('sha256').update(bytes).digest('hex');
              verificationMs += performance.now() - hashStart; recordPiece(current, identity);
              const start = Math.max(offset, current.offset), end = Math.min(offset + rangeLength, current.offset + current.size);
              const copyStart = performance.now();
              output.set(bytes.subarray(start - current.offset, end - current.offset), start - offset);
              genericCopyMs += performance.now() - copyStart; genericRequestedBytes += end - start;
            }
          }
          destination += rangeLength;
        }
      } finally { scratch = null; activeGenericScratchBytes = 0; }
      // Full and partial single leases also retain their admitted binding until
      // publication. A matching-content path replacement during hashing must not
      // relabel the old descriptor as the approved source. Empty reads own no FD.
      for (const path of new Set(ranges.filter(range => range.length > 0).map(range => range.path))) {
        const handle = handles.get(path);
        if (handle === undefined) throw new Error('Gather source descriptor missing.');
        assertIdentity(path, handle);
      }
    }
    if (closing) throw new Error('Gather worker closed.');
    if (batch) { batchReads++; batchRanges += ranges.length; batchBytes += length; }
    const timings = snapshotVerifiedPieceJobTimings({ workerWallMs: performance.now() - started,
      readWaitMs: acquisitionMs - before.acquisitionMs, verificationMs: verificationMs - before.verificationMs,
      copyMs: genericCopyMs - before.genericCopyMs });
    port.postMessage({ buffer, length, timings }, [buffer]);
  } catch (error) {
    // Never publish a verified prefix after a later piece fails verification.
    new Uint8Array(buffer).fill(0);
    port.postMessage({ buffer, error: String(error.message ?? error) }, [buffer]);
  }
}
port.on('message', message => {
  if (message.close === true) {
    closeWorker().then(() => port.postMessage({ closed: true, receipt: getReceipt() }),
      error => port.postMessage({ closed: true, error: String(error.message ?? error) }));
    return;
  }
  if (message.receipt === true) { port.postMessage({ receipt: getReceipt() }); return; }
  const operation = gather(message);
  operations.add(operation);
  operation.finally(() => operations.delete(operation)).catch(() => {});
});
process.on('exit', () => { for (const handle of handles.values()) closeSync(handle); });
