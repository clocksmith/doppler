import { parentPort, threadId, workerData } from 'node:worker_threads';
import {
  openSync, readSync, read as readAsync, writeSync, closeSync, fstatSync, lstatSync,
  futimesSync, readdirSync, readFileSync, renameSync, unlinkSync, constants,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { createVerifiedPieceReader } from './verified-piece-reader.js';
import { matchesFileStateSnapshot } from './verified-piece-state-cache.js';
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
 * @param {Uint8Array} bytes @param {number | null} [admittedHandle] */
function readFile(piece, bytes, admittedHandle = null) {
  const handle = admittedHandle ?? admitFile(piece, bytes);
  let read = 0;
  while (read < bytes.length) {
    const count = readSync(handle, bytes, read, bytes.length - read, piece.offset + read);
    if (!count) throw new Error('Incomplete authenticated piece.');
    read += count;
  }
  if (admittedHandle === null) assertIdentity(piece.path, handle);
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
let protectedCacheHits = 0, protectedCacheHitBytes = 0, protectedCacheMisses = 0;
let protectedCacheAdmissions = 0, protectedCacheAdmittedBytes = 0;
let protectedCacheEvictions = 0, protectedCacheEvictedBytes = 0, protectedCacheSkips = 0;
let stateCacheHits = 0, stateCacheHitBytes = 0, stateCacheMisses = 0;
let stateCacheAdmissions = 0, stateCacheAdmittedBytes = 0;
let backgroundVerifiedPieces = 0, backgroundVerificationMs = 0;
let pendingDirtyState = false;
const protectedCache = workerData.protectedCache ?? null;
const stateCacheConfig = workerData.stateCache ?? null;
const parsedIndex = JSON.parse(new TextDecoder().decode(workerData.metadata.indexBytes));
/** @type {Map<string, { canonicalPath: string, snapshot: import('./verified-piece-state-cache.js').FileStateSnapshot, wordCount: number, words: Int32Array, pieces: Array<{ offset: number, size: number, identity: string }> }>} */
const stateFiles = new Map();
if (stateCacheConfig?.files) {
  for (const file of parsedIndex.files) {
    const desc = stateCacheConfig.files[file.path];
    if (!desc) continue;
    stateFiles.set(file.path, {
      canonicalPath: desc.canonicalPath,
      snapshot: desc.snapshot,
      wordCount: desc.wordCount,
      words: new Int32Array(desc.sharedBuffer),
      pieces: file.pieces,
    });
  }
}

/** @param {Array<{ offset: number, size: number }>} pieces @param {number} offset */
function findStatePieceIndex(pieces, offset) {
  let low = 0, high = pieces.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    const p = pieces[mid];
    if (p.offset + p.size <= offset) low = mid + 1;
    else high = mid;
  }
  return low < pieces.length && pieces[low].offset === offset ? low : -1;
}

/** @param {string} path @param {number} handle */
function checkStateStat(path, handle) {
  const entry = stateFiles.get(path);
  if (!entry || Atomics.load(entry.words, entry.wordCount + 1) !== 0) return false;
  try {
    const fst = fstatSync(handle, { bigint: true });
    const lst = lstatSync(entry.canonicalPath, { bigint: true });
    if (!matchesFileStateSnapshot(fst, entry.snapshot) || !matchesFileStateSnapshot(lst, entry.snapshot)) {
      Atomics.store(entry.words, entry.wordCount + 1, 1);
      return false;
    }
    return true;
  } catch {
    Atomics.store(entry.words, entry.wordCount + 1, 1);
    return false;
  }
}

/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece */
function isStateVerified(piece) {
  const entry = stateFiles.get(piece.path);
  if (!entry || Atomics.load(entry.words, entry.wordCount + 1) !== 0) return false;
  const idx = findStatePieceIndex(entry.pieces, piece.offset);
  return idx >= 0 && (Atomics.load(entry.words, idx >>> 5) & (1 << (idx & 31))) !== 0;
}

/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece */
function markStateVerified(piece) {
  const entry = stateFiles.get(piece.path);
  if (!entry || Atomics.load(entry.words, entry.wordCount + 1) !== 0) return false;
  const idx = findStatePieceIndex(entry.pieces, piece.offset);
  if (idx < 0) return false;
  const mask = 1 << (idx & 31);
  const prev = Atomics.or(entry.words, idx >>> 5, mask);
  if ((prev & mask) === 0) {
    Atomics.add(entry.words, entry.wordCount, 1);
    stateCacheAdmissions++;
    stateCacheAdmittedBytes += piece.size;
    pendingDirtyState = true;
    return true;
  }
  return false;
}
const bucketBytesView = protectedCache ? new BigInt64Array(protectedCache.sharedBucketBytes) : null;
const expectedUid = typeof process.geteuid === 'function' ? process.geteuid() : null;
let tmpSeq = 0, lastMemCheckAt = 0, lastMemAvailableBytes = Infinity;

/** @param {number} extraBytes */
function hasHostMemoryReserve(extraBytes) {
  if (!protectedCache || protectedCache.minHostAvailableBytes <= 0 || process.platform !== 'linux') return true;
  const now = performance.now();
  if (now - lastMemCheckAt >= 50) {
    lastMemCheckAt = now;
    try {
      const text = readFileSync('/proc/meminfo', 'utf8');
      const match = /^MemAvailable:\s+(\d+)/mu.exec(text);
      lastMemAvailableBytes = match ? Number(match[1]) * 1024 : 0;
    } catch {
      lastMemAvailableBytes = 0;
    }
  }
  return lastMemAvailableBytes - extraBytes >= protectedCache.minHostAvailableBytes;
}

/** @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece */
function protectedLocation(piece) {
  if (!protectedCache || typeof piece.identity !== 'string' || !piece.identity.startsWith('sha256:')) return null;
  const hex = piece.identity.slice(7);
  if (!/^[a-f0-9]{64}$/u.test(hex)) return null;
  const bucketIndex = parseInt(hex.slice(0, 2), 16) & (protectedCache.bucketCount - 1);
  const bucketDir = join(protectedCache.directory, bucketIndex.toString(16).padStart(2, '0'));
  return { hex, bucketIndex, bucketDir, cachePath: join(bucketDir, `${hex}.piece`) };
}

/**
 * Attempt to read a complete piece from the protected cache, authenticating its
 * full SHA-256 digest before accepting. Corrupted cache entries are unlinked and
 * fall back transparently to the admitted underlying source file.
 * @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} output
 * @param {boolean} isDirect
 */
function tryReadProtectedPiece(piece, output, isDirect) {
  const loc = protectedLocation(piece);
  if (!loc || output.byteLength !== piece.size) return false;
  let fd = -1;
  const readStart = performance.now();
  try {
    fd = openSync(loc.cachePath, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const st = fstatSync(fd);
    if (!st.isFile() || st.size !== piece.size || (expectedUid !== null && st.uid !== expectedUid)) {
      try { unlinkSync(loc.cachePath); } catch {}
      return false;
    }
    let read = 0;
    while (read < output.length) {
      const count = readSync(fd, output, read, output.length - read, read);
      if (!count) {
        try { unlinkSync(loc.cachePath); } catch {}
        return false;
      }
      read += count;
    }
    if (Date.now() - st.mtimeMs >= 5000) {
      const nowSec = Date.now() / 1000;
      try { futimesSync(fd, nowSec, nowSec); } catch {}
    }
  } catch {
    return false;
  } finally {
    if (fd !== -1) {
      try { closeSync(fd); } catch {}
    }
    acquisitionMs += performance.now() - readStart;
  }
  const hashStart = performance.now();
  const digest = 'sha256:' + createHash('sha256').update(output).digest('hex');
  verificationMs += performance.now() - hashStart;
  if (digest !== piece.identity) {
    output.fill(0);
    try {
      unlinkSync(loc.cachePath);
      if (bucketBytesView) Atomics.sub(bucketBytesView, loc.bucketIndex, BigInt(piece.size));
    } catch {}
    return false;
  }
  if (isDirect) recordDirect(piece, digest);
  else recordPiece(piece, digest);
  protectedCacheHits++;
  protectedCacheHitBytes += piece.size;
  return true;
}

/**
 * Admit an already SHA-256-authenticated piece into the bounded protected cache.
 * Uses scan-resistant hash-priority retention among fresh entries and evicts
 * stale entries across workload transitions without exceeding bucketMaxBytes.
 * @param {Readonly<import('./verified-piece-reader.js').VerifiedPiece>} piece
 * @param {Uint8Array} output
 */
function admitProtectedPiece(piece, output) {
  const loc = protectedLocation(piece);
  if (!loc || !protectedCache || !bucketBytesView || closing
    || output.byteLength !== piece.size || piece.size > protectedCache.bucketMaxBytes) {
    return;
  }
  if (!hasHostMemoryReserve(piece.size)) {
    protectedCacheSkips++;
    return;
  }
  const sizeBig = BigInt(piece.size);
  const maxBig = BigInt(protectedCache.bucketMaxBytes);
  if (Atomics.load(bucketBytesView, loc.bucketIndex) + sizeBig > maxBig) {
    const nowMs = Date.now();
    let actualBytes = 0;
    /** @type {Array<{ hex: string, path: string, size: number, mtimeMs: number }>} */
    const stale = [];
    /** @type {Array<{ hex: string, path: string, size: number, mtimeMs: number }>} */
    const lowerPriorityFresh = [];
    try {
      for (const name of readdirSync(loc.bucketDir)) {
        const entryPath = join(loc.bucketDir, name);
        if (name.includes('.tmp.')) {
          try {
            const st = lstatSync(entryPath);
            if (nowMs - st.mtimeMs > 30_000) unlinkSync(entryPath);
          } catch {}
          continue;
        }
        if (!name.endsWith('.piece')) continue;
        try {
          const st = lstatSync(entryPath);
          if (!st.isFile()) continue;
          const entryHex = name.slice(0, -6);
          if (entryHex === loc.hex && st.size === piece.size) return;
          actualBytes += st.size;
          if (nowMs - st.mtimeMs > protectedCache.staleAfterMs) {
            stale.push({ hex: entryHex, path: entryPath, size: st.size, mtimeMs: st.mtimeMs });
          } else if (entryHex > loc.hex) {
            lowerPriorityFresh.push({ hex: entryHex, path: entryPath, size: st.size, mtimeMs: st.mtimeMs });
          }
        } catch {}
      }
    } catch {
      protectedCacheSkips++;
      return;
    }
    Atomics.store(bucketBytesView, loc.bucketIndex, BigInt(actualBytes));
    if (actualBytes + piece.size > protectedCache.bucketMaxBytes) {
      stale.sort((a, b) => a.mtimeMs - b.mtimeMs);
      lowerPriorityFresh.sort((a, b) => (a.hex < b.hex ? 1 : a.hex > b.hex ? -1 : 0));
      const candidates = [...stale, ...lowerPriorityFresh];
      const reclaimable = candidates.reduce((sum, item) => sum + item.size, 0);
      if (actualBytes - reclaimable + piece.size > protectedCache.bucketMaxBytes) {
        protectedCacheSkips++;
        return;
      }
      for (const victim of candidates) {
        if (actualBytes + piece.size <= protectedCache.bucketMaxBytes) break;
        try {
          unlinkSync(victim.path);
          actualBytes -= victim.size;
          Atomics.sub(bucketBytesView, loc.bucketIndex, BigInt(victim.size));
          protectedCacheEvictions++;
          protectedCacheEvictedBytes += victim.size;
        } catch {}
      }
      if (actualBytes + piece.size > protectedCache.bucketMaxBytes) {
        protectedCacheSkips++;
        return;
      }
    }
  }
  const reservedAfter = Atomics.add(bucketBytesView, loc.bucketIndex, sizeBig) + sizeBig;
  if (reservedAfter > maxBig) {
    Atomics.sub(bucketBytesView, loc.bucketIndex, sizeBig);
    protectedCacheSkips++;
    return;
  }
  const tmpPath = join(loc.bucketDir, `${loc.hex}.tmp.${process.pid}.${threadId}.${++tmpSeq}`);
  let fd = -1;
  try {
    try {
      const existing = lstatSync(loc.cachePath);
      if (existing.isFile() && existing.size === piece.size) {
        Atomics.sub(bucketBytesView, loc.bucketIndex, sizeBig);
        return;
      }
    } catch {}
    fd = openSync(tmpPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW, 0o600);
    let written = 0;
    while (written < output.length) {
      written += writeSync(fd, output, written, output.length - written, written);
    }
    closeSync(fd);
    fd = -1;
    renameSync(tmpPath, loc.cachePath);
    protectedCacheAdmissions++;
    protectedCacheAdmittedBytes += piece.size;
  } catch {
    if (fd !== -1) {
      try { closeSync(fd); } catch {}
    }
    try { unlinkSync(tmpPath); } catch {}
    Atomics.sub(bucketBytesView, loc.bucketIndex, sizeBig);
    protectedCacheSkips++;
  }
}
/** @type {Map<string, Readonly<import('./verified-piece-reader.js').VerifiedPiece>>} */
const verifiedRanges = new Map();
const maxPieceBytes = parsedIndex.maxPieceBytes;
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
  const ranges = new Map([
    ...generic.ranges.map(range => /** @type {const} */ ([`${range.path}:${range.offset}`, range])),
    ...verifiedRanges,
  ]);
  const pieces = new Map([...ranges.values()].map(range => [range.identity, range.size]));
  return { ...generic, ranges: [...ranges.values()], pieces: [...pieces.keys()],
    verifiedBytes: [...pieces.values()].reduce((sum, size) => sum + size, 0),
    requestedBytes: generic.requestedBytes + directPieceBytes + genericRequestedBytes,
    copyMs: generic.copyMs + genericCopyMs,
    acquisitionMs: generic.acquisitionMs + acquisitionMs, verificationMs: generic.verificationMs + verificationMs,
    directPieceReads, directPieceBytes, batchReads, batchRanges, batchBytes,
    activeDigestCopyBytes: 0, peakDigestCopyBytes: 0,
    activeGenericScratchBytes, peakGenericScratchBytes,
    protectedCacheHits, protectedCacheHitBytes, protectedCacheMisses,
    protectedCacheAdmissions, protectedCacheAdmittedBytes,
    protectedCacheEvictions, protectedCacheEvictedBytes, protectedCacheSkips,
    stateCacheHits, stateCacheHitBytes, stateCacheMisses,
    stateCacheAdmissions, stateCacheAdmittedBytes,
    backgroundVerifiedPieces, backgroundVerificationMs };
}
port.postMessage({ ready: true });
/** @type {Set<Promise<void>>} */
const operations = new Set();
/** @type {ReturnType<typeof setTimeout> | null} */
let bgTimer = null;
/** @type {Promise<void> | null} */
let bgActive = null;
/** @type {Array<Readonly<import('./verified-piece-reader.js').VerifiedPiece>>} */
const bgQueue = [];
let bgCursor = 0;
/** @type {Uint8Array | null} */
let bgScratch = null;
if (stateCacheConfig?.backgroundVerify === true && stateFiles.size > 0) {
  const workerIndex = workerData.workerIndex ?? 0;
  const workerCount = Math.max(1, workerData.workerCount ?? 1);
  let seq = 0;
  for (const [piecePath, entry] of stateFiles) {
    const prio = new Set(stateCacheConfig.files?.[piecePath]?.priorityPieceIndexes ?? []);
    const ordered = prio.size ? [...entry.pieces.filter((_, i) => prio.has(i)), ...entry.pieces.filter((_, i) => !prio.has(i))] : entry.pieces;
    for (const piece of ordered) { if (seq++ % workerCount === workerIndex) bgQueue.push(Object.freeze({ ...piece, path: piecePath })); }
  }
}

function scheduleBackgroundVerify() {
  if (closing || !bgQueue.length || bgCursor >= bgQueue.length || bgTimer !== null || bgActive !== null) return;
  bgTimer = setTimeout(() => {
    bgTimer = null;
    if (closing || operations.size > 0 || bgActive !== null) return;
    while (bgCursor < bgQueue.length && isStateVerified(bgQueue[bgCursor])) bgCursor++;
    if (bgCursor >= bgQueue.length) { bgScratch = null; return; }
    const piece = bgQueue[bgCursor++];
    const task = (async () => {
      bgScratch ??= new Uint8Array(maxPieceBytes);
      const view = bgScratch.subarray(0, piece.size);
      const handle = admitFile(piece, view);
      if (!checkStateStat(piece.path, handle) || isStateVerified(piece)) return;
      await readFileParallel(piece, view, handle);
      if (closing || operations.size > 0 || !checkStateStat(piece.path, handle)) return;
      const hashStart = performance.now();
      const digest = 'sha256:' + createHash('sha256').update(view).digest('hex');
      backgroundVerificationMs += performance.now() - hashStart;
      if (digest === piece.identity && checkStateStat(piece.path, handle)) {
        markStateVerified(piece);
        backgroundVerifiedPieces++;
        if (pendingDirtyState) {
          pendingDirtyState = false;
          port.postMessage({ dirtyState: true });
        }
      }
    })().catch(() => {}).finally(() => {
      bgActive = null;
      if (!closing && operations.size === 0 && bgCursor < bgQueue.length) scheduleBackgroundVerify();
    });
    bgActive = task;
  }, stateCacheConfig?.backgroundIdleDelayMs ?? 50);
  bgTimer.unref?.();
}
if (bgQueue.length > 0) scheduleBackgroundVerify();

/** @type {Promise<void> | null} */
let retirement = null;
function closeWorker() {
  closing = true;
  if (bgTimer !== null) {
    clearTimeout(bgTimer);
    bgTimer = null;
  }
  if (!retirement) {
    const operation = (async () => {
      if (bgActive) await bgActive;
      bgScratch = null;
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
        const handle = /** @type {number} */ (admitted.get(piece.path));
        return (async () => {
          const stateOkBefore = checkStateStat(piece.path, handle);
          if (stateOkBefore && isStateVerified(piece)) {
            const readStart = performance.now();
            try {
              if (pieces.length === 1) readFile(piece, output, handle);
              else await readFileParallel(piece, output, handle);
            } finally { acquisitionMs += performance.now() - readStart; }
            if (closing) throw new Error('Gather worker closed.');
            if (checkStateStat(piece.path, handle)) {
              recordDirect(piece, piece.identity);
              stateCacheHits++;
              stateCacheHitBytes += piece.size;
              return;
            }
          }
          if (stateFiles.has(piece.path)) stateCacheMisses++;
          if (tryReadProtectedPiece(piece, output, true)) {
            if (closing) throw new Error('Gather worker closed.');
            // The authenticated bytes came from the protected copy. The file's own bytes at this piece were
            // not read, so the file is not marked verified: a direct read of the file must still be refused.
            return;
          }
          if (protectedCache) protectedCacheMisses++;
          const readStart = performance.now();
          try {
            if (pieces.length === 1) readFile(piece, output, handle);
            else await readFileParallel(piece, output, handle);
          } finally { acquisitionMs += performance.now() - readStart; }
          if (closing) throw new Error('Gather worker closed.');
          // This worker exclusively owns the output until transfer. Hash that
          // storage directly, without WebCrypto's input snapshot and another
          // shared thread-pool handoff. Other worker lanes still run in parallel.
          authenticateDirect(piece, output);
          if (stateOkBefore && checkStateStat(piece.path, handle)) markStateVerified(piece);
          else admitProtectedPiece(piece, output);
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
            const admittedHandle = admitFile(piece, output);
            const stateOkBefore = checkStateStat(piece.path, admittedHandle);
            let satisfied = false;
            if (stateOkBefore && isStateVerified(piece)) {
              const readStart = performance.now();
              try { readFile(piece, output, admittedHandle); }
              finally { acquisitionMs += performance.now() - readStart; }
              if (checkStateStat(piece.path, admittedHandle)) {
                recordDirect(piece, piece.identity);
                stateCacheHits++;
                stateCacheHitBytes += piece.size;
                satisfied = true;
              } else {
                authenticateDirect(piece, output);
                admitProtectedPiece(piece, output);
                satisfied = true;
              }
            }
            if (!satisfied) {
              if (stateFiles.has(piece.path)) stateCacheMisses++;
              if (!tryReadProtectedPiece(piece, output, true)) {
                if (protectedCache) protectedCacheMisses++;
                const readStart = performance.now();
                try { readFile(piece, output, admittedHandle); }
                finally { acquisitionMs += performance.now() - readStart; }
                authenticateDirect(piece, output);
                if (stateOkBefore && checkStateStat(piece.path, admittedHandle)) markStateVerified(piece);
                else admitProtectedPiece(piece, output);
              }
            }
          } else {
            const selected = reader.resolveRangePieces(path, offset, rangeLength);
            if (!selected.length) { destination += rangeLength; continue; }
            if (!scratch) {
              scratch = new Uint8Array(maxPieceBytes);
              activeGenericScratchBytes = scratch.byteLength;
              peakGenericScratchBytes = Math.max(peakGenericScratchBytes, activeGenericScratchBytes);
            }
            for (const current of selected) {
              const bytes = scratch.subarray(0, current.size);
              const admittedHandle = admitFile(current, bytes);
              const stateOkBefore = checkStateStat(current.path, admittedHandle);
              let satisfied = false;
              if (stateOkBefore && isStateVerified(current)) {
                const readStart = performance.now();
                try { readFile(current, bytes, admittedHandle); }
                finally { acquisitionMs += performance.now() - readStart; }
                if (checkStateStat(current.path, admittedHandle)) {
                  recordPiece(current, current.identity);
                  stateCacheHits++;
                  stateCacheHitBytes += current.size;
                  satisfied = true;
                } else {
                  const hashStart = performance.now();
                  const identity = 'sha256:' + createHash('sha256').update(bytes).digest('hex');
                  verificationMs += performance.now() - hashStart;
                  recordPiece(current, identity);
                  admitProtectedPiece(current, bytes);
                  satisfied = true;
                }
              }
              if (!satisfied) {
                if (stateFiles.has(current.path)) stateCacheMisses++;
                if (!tryReadProtectedPiece(current, bytes, false)) {
                  if (protectedCache) protectedCacheMisses++;
                  const readStart = performance.now();
                  try { readFile(current, bytes, admittedHandle); }
                  finally { acquisitionMs += performance.now() - readStart; }
                  const hashStart = performance.now();
                  const identity = 'sha256:' + createHash('sha256').update(bytes).digest('hex');
                  verificationMs += performance.now() - hashStart; recordPiece(current, identity);
                  if (stateOkBefore && checkStateStat(current.path, admittedHandle)) markStateVerified(current);
                  else admitProtectedPiece(current, bytes);
                }
              }
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
    const dirtyState = pendingDirtyState;
    pendingDirtyState = false;
    const timings = snapshotVerifiedPieceJobTimings({ workerWallMs: performance.now() - started,
      readWaitMs: acquisitionMs - before.acquisitionMs, verificationMs: verificationMs - before.verificationMs,
      copyMs: genericCopyMs - before.genericCopyMs });
    port.postMessage({ buffer, length, timings, ...(dirtyState ? { dirtyState: true } : {}) }, [buffer]);
  } catch (error) {
    // Never publish a verified prefix after a later piece fails verification.
    new Uint8Array(buffer).fill(0);
    port.postMessage({ buffer, error: String(error.message ?? error) }, [buffer]);
  } finally {
    if (!closing && bgQueue.length > 0 && bgCursor < bgQueue.length) scheduleBackgroundVerify();
  }
}
port.on('message', message => {
  if (bgTimer !== null) {
    clearTimeout(bgTimer);
    bgTimer = null;
  }
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
