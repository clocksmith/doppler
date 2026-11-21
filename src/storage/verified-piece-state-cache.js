import { createHash } from 'node:crypto';
import {
  closeSync, fstatSync, lstatSync, mkdirSync, openSync, readSync,
  realpathSync, renameSync, unlinkSync, writeSync, constants,
} from 'node:fs';
import { open, rename, unlink } from 'node:fs/promises';
import { isAbsolute, join } from 'node:path';

const FILE_STATE_SCHEMA = 'doppler.verified-piece-file-state/v1';
const PIECE_DIR_STATE_SCHEMA = 'doppler.verified-piece-dir-state/v1';
const HEADER_BYTES = 32;

/** @param {string | Uint8Array} input */
const sha256Bytes = input => createHash('sha256').update(input).digest();

/** @param {string | Uint8Array} input */
const sha256Hex = input => createHash('sha256').update(input).digest('hex');

/**
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 */
function equalBytes(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * @param {string} directory
 * @param {number | null} expectedUid
 */
function ensureOwnedStateDirectory(directory, expectedUid) {
  if (typeof directory !== 'string' || !directory.trim() || !isAbsolute(directory)) {
    throw new Error('Piece state verification cache requires an absolute directory.');
  }
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const st = lstatSync(directory);
  if (!st.isDirectory() || (st.mode & 0o022) !== 0 || (expectedUid !== null && st.uid !== expectedUid)) {
    throw new Error('Piece state verification cache directory must be an owned non-symlink directory.');
  }
  return realpathSync(directory);
}

/**
 * @param {Array<{ offset: number, size: number }>} pieces
 * @param {number} offset
 */
function findPieceIndex(pieces, offset) {
  let low = 0;
  let high = pieces.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    const piece = pieces[mid];
    if (piece.offset + piece.size <= offset) low = mid + 1;
    else high = mid;
  }
  return low < pieces.length && pieces[low].offset === offset ? low : -1;
}

/**
 * @param {string} stateFilePath
 * @param {number} payloadBytes
 * @param {number | null} expectedUid
 */
function tryLoadVerifiedPayload(stateFilePath, payloadBytes, expectedUid) {
  let fd = -1;
  try {
    fd = openSync(stateFilePath, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const st = fstatSync(fd);
    if (!st.isFile() || (st.mode & 0o022) !== 0 || st.size !== HEADER_BYTES + payloadBytes
      || (expectedUid !== null && st.uid !== expectedUid)) {
      try { unlinkSync(stateFilePath); } catch {}
      return null;
    }
    const raw = new Uint8Array(HEADER_BYTES + payloadBytes);
    let read = 0;
    while (read < raw.byteLength) {
      const count = readSync(fd, raw, read, raw.byteLength - read, read);
      if (!count) {
        try { unlinkSync(stateFilePath); } catch {}
        return null;
      }
      read += count;
    }
    const header = raw.subarray(0, HEADER_BYTES);
    const payload = raw.subarray(HEADER_BYTES);
    if (!equalBytes(header, sha256Bytes(payload))) {
      try { unlinkSync(stateFilePath); } catch {}
      return null;
    }
    return payload;
  } catch {
    return null;
  } finally {
    if (fd !== -1) {
      try { closeSync(fd); } catch {}
    }
  }
}

let tempCounter = 0;

/**
 * @param {string} stateFilePath
 * @param {Uint8Array} payloadSnapshot
 */
async function writeVerifiedPayloadAsync(stateFilePath, payloadSnapshot) {
  const tmpPath = `${stateFilePath}.tmp.${process.pid}.${++tempCounter}`;
  const out = new Uint8Array(HEADER_BYTES + payloadSnapshot.byteLength);
  out.set(sha256Bytes(payloadSnapshot), 0);
  out.set(payloadSnapshot, HEADER_BYTES);
  let handle = null;
  try {
    handle = await open(tmpPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW, 0o600);
    let written = 0;
    while (written < out.byteLength) {
      const res = await handle.write(out, written, out.byteLength - written, written);
      if (!res.bytesWritten) throw new Error('Incomplete state cache write.');
      written += res.bytesWritten;
    }
    await handle.close();
    handle = null;
    await rename(tmpPath, stateFilePath);
  } catch {
    if (handle) {
      try { await handle.close(); } catch {}
    }
    try { await unlink(tmpPath); } catch {}
  }
}

/**
 * @param {string} stateFilePath
 * @param {Uint8Array} payloadSnapshot
 */
function writeVerifiedPayloadSync(stateFilePath, payloadSnapshot) {
  const tmpPath = `${stateFilePath}.tmp.${process.pid}.${++tempCounter}`;
  const out = new Uint8Array(HEADER_BYTES + payloadSnapshot.byteLength);
  out.set(sha256Bytes(payloadSnapshot), 0);
  out.set(payloadSnapshot, HEADER_BYTES);
  let fd = -1;
  try {
    fd = openSync(tmpPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW, 0o600);
    let written = 0;
    while (written < out.byteLength) {
      written += writeSync(fd, out, written, out.byteLength - written, written);
    }
    closeSync(fd);
    fd = -1;
    renameSync(tmpPath, stateFilePath);
  } catch {
    if (fd !== -1) {
      try { closeSync(fd); } catch {}
    }
    try { unlinkSync(tmpPath); } catch {}
  }
}

/**
 * @param {import('node:fs').BigIntStats} stat
 * @param {{ device: string, inode: string, size: string, mtimeNs: string, ctimeNs: string }} snapshot
 */
export function matchesFileStateSnapshot(stat, snapshot) {
  return stat.isFile()
    && stat.dev.toString() === snapshot.device
    && stat.ino.toString() === snapshot.inode
    && stat.size.toString() === snapshot.size
    && stat.mtimeNs.toString() === snapshot.mtimeNs
    && stat.ctimeNs.toString() === snapshot.ctimeNs;
}

/**
 * Prepare shared in-memory + optional persistent per-file-state verification bitsets
 * across main-thread readers and worker gather lanes.
 * @type {typeof import('./verified-piece-state-cache.js').prepareVerifiedPieceStateCache}
 */
export async function prepareVerifiedPieceStateCache(options) {
  const { rawPolicy, manifestIdentity, indexIdentity, manifestBytes, index, files } = options;
  if (rawPolicy === undefined) return undefined;
  if (!rawPolicy || typeof rawPolicy !== 'object' || Array.isArray(rawPolicy)) {
    throw new Error('Piece state verification cache policy must be an explicit object.');
  }
  const backgroundVerify = rawPolicy.backgroundVerify === true;
  const backgroundIdleDelayMs = rawPolicy.backgroundIdleDelayMs ?? 50;
  const rawLayerRange = rawPolicy.priorityLayerRange ?? null;
  if (typeof backgroundVerify !== 'boolean'
    || !Number.isSafeInteger(backgroundIdleDelayMs) || backgroundIdleDelayMs < 0 || backgroundIdleDelayMs > 60_000
    || (rawLayerRange !== null && (!Array.isArray(rawLayerRange) || rawLayerRange.length !== 2
      || !Number.isSafeInteger(rawLayerRange[0]) || !Number.isSafeInteger(rawLayerRange[1])
      || rawLayerRange[0] < 0 || rawLayerRange[1] < rawLayerRange[0]))) {
    throw new Error('Piece state verification cache requires valid backgroundVerify, backgroundIdleDelayMs, and optional priorityLayerRange.');
  }
  /** @type {Map<string, Array<[number, number]>>} */
  const expertRangesByPath = new Map();
  if (backgroundVerify && manifestBytes instanceof Uint8Array && manifestBytes.byteLength > 0) {
    try {
      const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
      const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
      const tensors = manifest?.tensors && typeof manifest.tensors === 'object' ? Object.entries(manifest.tensors) : [];
      for (const [name, loc] of tensors) {
        if (!/(?:\.mlp\.experts\.|_exps\.weight$)/u.test(name) || !loc || typeof loc !== 'object') continue;
        if (rawLayerRange !== null) {
          const match = /(?:layers|blk)\.(\d+)\./u.exec(name);
          const layerIdx = match ? Number(match[1]) : -1;
          if (layerIdx < rawLayerRange[0] || layerIdx >= rawLayerRange[1]) continue;
        }
        const spans = Array.isArray(loc.spans) && loc.spans.length > 0
          ? loc.spans : [{ shard: loc.shard ?? loc.shardIndex, offset: loc.offset, size: loc.size }];
        for (const span of spans) {
          const shardIdx = span?.shard ?? span?.shardIndex;
          const filename = Number.isSafeInteger(shardIdx) && shardIdx >= 0 ? shards[shardIdx]?.filename : null;
          if (typeof filename !== 'string' || !Number.isSafeInteger(span?.offset) || !Number.isSafeInteger(span?.size) || span.size <= 0) continue;
          let list = expertRangesByPath.get(filename);
          if (!list) { list = []; expertRangesByPath.set(filename, list); }
          list.push([span.offset, span.offset + span.size]);
        }
      }
    } catch { /* Malformed metadata is rejected by the piece reader. */ }
  }
  const expectedUid = typeof process.geteuid === 'function' ? process.geteuid() : null;
  const directory = rawPolicy.directory === undefined
    ? null
    : ensureOwnedStateDirectory(rawPolicy.directory, expectedUid);

  /** @type {Record<string, import('./verified-piece-state-cache.js').WorkerFileStateDescriptor>} */
  const workerFiles = {};
  /** @type {Map<string, { pieces: Array<{ offset: number, size: number }>, words: Int32Array, wordCount: number, snapshot: import('./verified-piece-state-cache.js').FileStateSnapshot, canonicalPath: string, stateFilePath: string | null, lastSavedSeq: number }>} */
  const entries = new Map();

  for (const file of index.files) {
    const mapped = files?.[file.path];
    if (typeof mapped !== 'string') continue;
    let st;
    try {
      st = lstatSync(mapped, { bigint: true });
    } catch {
      continue;
    }
    if (!st.isFile() || st.size !== BigInt(file.size)) continue;
    const snapshot = {
      device: st.dev.toString(),
      inode: st.ino.toString(),
      size: st.size.toString(),
      mtimeNs: st.mtimeNs.toString(),
      ctimeNs: st.ctimeNs.toString(),
    };
    const wordCount = Math.max(1, Math.ceil(file.pieces.length / 32));
    const sharedBuffer = new SharedArrayBuffer((wordCount + 2) * 4);
    const words = new Int32Array(sharedBuffer);
    let stateFilePath = null;
    if (directory !== null) {
      const stateKey = sha256Hex(JSON.stringify({
        schema: FILE_STATE_SCHEMA,
        manifestIdentity,
        indexIdentity,
        piecePath: file.path,
        canonicalPath: mapped,
        uid: expectedUid,
        ...snapshot,
        pieceCount: file.pieces.length,
      }));
      stateFilePath = join(directory, `${stateKey}.state`);
      const loaded = tryLoadVerifiedPayload(stateFilePath, wordCount * 4, expectedUid);
      if (loaded) {
        new Uint8Array(sharedBuffer, 0, wordCount * 4).set(loaded);
      }
    }
    const spans = expertRangesByPath.get(file.path) ?? null;
    const priorityPieceIndexes = spans
      ? file.pieces.flatMap((piece, idx) => spans.some(([s, e]) => piece.offset < e && piece.offset + piece.size > s) ? [idx] : [])
      : undefined;
    workerFiles[file.path] = {
      canonicalPath: mapped,
      snapshot,
      wordCount,
      pieceCount: file.pieces.length,
      sharedBuffer,
      stateFilePath,
      ...(priorityPieceIndexes?.length ? { priorityPieceIndexes } : {}),
    };
    entries.set(file.path, {
      pieces: file.pieces,
      words,
      wordCount,
      snapshot,
      canonicalPath: mapped,
      stateFilePath,
      lastSavedSeq: 0,
    });
  }

  /** @type {ReturnType<typeof setTimeout> | null} */
  let flushTimer = null;
  /** @type {Promise<void>} */
  let flushChain = Promise.resolve();
  let closed = false;

  const flushDirty = () => {
    if (directory === null) return Promise.resolve();
    flushChain = flushChain.then(async () => {
      for (const entry of entries.values()) {
        if (!entry.stateFilePath) continue;
        if (Atomics.load(entry.words, entry.wordCount + 1) !== 0) continue;
        const seq = Atomics.load(entry.words, entry.wordCount);
        if (seq === entry.lastSavedSeq) continue;
        const snapshotBytes = new Uint8Array(entry.wordCount * 4);
        for (let w = 0; w < entry.wordCount; w++) {
          const val = Atomics.load(entry.words, w);
          snapshotBytes[w * 4] = val & 0xff;
          snapshotBytes[w * 4 + 1] = (val >>> 8) & 0xff;
          snapshotBytes[w * 4 + 2] = (val >>> 16) & 0xff;
          snapshotBytes[w * 4 + 3] = (val >>> 24) & 0xff;
        }
        await writeVerifiedPayloadAsync(entry.stateFilePath, snapshotBytes);
        entry.lastSavedSeq = seq;
      }
    }).catch(() => {});
    return flushChain;
  };

  const scheduleFlush = () => {
    if (directory === null || closed || flushTimer !== null) return;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      void flushDirty();
    }, 25);
    flushTimer.unref?.();
  };

  /**
   * Check whether a mapped full file currently matches its admitted snapshot,
   * invalidating cached bits immediately if any stat field changed.
   * @param {string} piecePath
   * @param {import('node:fs').BigIntStats} stat
   */
  const checkFileStat = (piecePath, stat) => {
    const entry = entries.get(piecePath);
    if (!entry) return false;
    if (!matchesFileStateSnapshot(stat, entry.snapshot)) {
      Atomics.store(entry.words, entry.wordCount + 1, 1);
      return false;
    }
    return Atomics.load(entry.words, entry.wordCount + 1) === 0;
  };

  /**
   * @param {string} piecePath
   * @param {number} offset
   */
  const isPieceVerified = (piecePath, offset) => {
    const entry = entries.get(piecePath);
    if (!entry || Atomics.load(entry.words, entry.wordCount + 1) !== 0) return false;
    const idx = findPieceIndex(entry.pieces, offset);
    if (idx < 0) return false;
    return (Atomics.load(entry.words, idx >>> 5) & (1 << (idx & 31))) !== 0;
  };

  /**
   * @param {string} piecePath
   * @param {number} offset
   */
  const markPieceVerified = (piecePath, offset) => {
    const entry = entries.get(piecePath);
    if (!entry || Atomics.load(entry.words, entry.wordCount + 1) !== 0) return false;
    const idx = findPieceIndex(entry.pieces, offset);
    if (idx < 0) return false;
    const mask = 1 << (idx & 31);
    const prev = Atomics.or(entry.words, idx >>> 5, mask);
    if ((prev & mask) === 0) {
      Atomics.add(entry.words, entry.wordCount, 1);
      scheduleFlush();
      return true;
    }
    return false;
  };

  const close = async () => {
    closed = true;
    if (flushTimer !== null) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    await flushDirty();
  };

  return Object.freeze({
    directory,
    backgroundVerify,
    backgroundIdleDelayMs,
    workerPayload: Object.freeze({
      directory,
      backgroundVerify,
      backgroundIdleDelayMs,
      files: workerFiles,
    }),
    checkFileStat,
    isPieceVerified,
    markPieceVerified,
    scheduleFlush,
    flushSync() {
      if (directory === null) return;
      for (const entry of entries.values()) {
        if (!entry.stateFilePath || Atomics.load(entry.words, entry.wordCount + 1) !== 0) continue;
        const seq = Atomics.load(entry.words, entry.wordCount);
        if (seq === entry.lastSavedSeq) continue;
        const snapshotBytes = new Uint8Array(entry.wordCount * 4);
        for (let w = 0; w < entry.wordCount; w++) {
          const val = Atomics.load(entry.words, w);
          snapshotBytes[w * 4] = val & 0xff;
          snapshotBytes[w * 4 + 1] = (val >>> 8) & 0xff;
          snapshotBytes[w * 4 + 2] = (val >>> 16) & 0xff;
          snapshotBytes[w * 4 + 3] = (val >>> 24) & 0xff;
        }
        writeVerifiedPayloadSync(entry.stateFilePath, snapshotBytes);
        entry.lastSavedSeq = seq;
      }
    },
    close,
  });
}

/**
 * Persistent per-piece-file `(dev, ino, size, mtimeNs, ctimeNs)` verification cache
 * for partial piece-cache directories (`<pieceDirectory>/<path>.<offset>`).
 * @type {typeof import('./verified-piece-state-cache.js').openPieceDirectoryStateCache}
 */
export async function openPieceDirectoryStateCache(options) {
  const { directory: rawDir, pieceDirectory, manifestIdentity, indexIdentity, index } = options;
  const expectedUid = typeof process.geteuid === 'function' ? process.geteuid() : null;
  const directory = ensureOwnedStateDirectory(rawDir, expectedUid);
  const canonicalPieceDir = realpathSync(pieceDirectory);
  const dirStat = lstatSync(canonicalPieceDir, { bigint: true });
  if (!dirStat.isDirectory()) throw new Error('Piece directory must be a non-symlink directory.');

  /** @type {Map<string, { baseIndex: number, pieces: Array<{ offset: number, size: number }> }>} */
  const fileLayout = new Map();
  let totalPieces = 0;
  for (const file of index.files) {
    fileLayout.set(file.path, { baseIndex: totalPieces, pieces: file.pieces });
    totalPieces += file.pieces.length;
  }

  // 4 x BigUint64 per piece: [dev, ino, mtimeNs, ctimeNs]; piece size is fixed in index.
  const payloadBytes = totalPieces * 32;
  const tableBuffer = new ArrayBuffer(payloadBytes);
  const table = new BigUint64Array(tableBuffer);
  const stateKey = sha256Hex(JSON.stringify({
    schema: PIECE_DIR_STATE_SCHEMA,
    manifestIdentity,
    indexIdentity,
    pieceDirectory: canonicalPieceDir,
    uid: expectedUid,
    totalPieces,
  }));
  const stateFilePath = join(directory, `${stateKey}.pstate`);
  const loaded = tryLoadVerifiedPayload(stateFilePath, payloadBytes, expectedUid);
  if (loaded) {
    new Uint8Array(tableBuffer).set(loaded);
  }

  let dirty = false;
  let savedDirtyGen = 0;
  let dirtyGen = 0;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let flushTimer = null;
  /** @type {Promise<void>} */
  let flushChain = Promise.resolve();
  let closed = false;

  /**
   * @param {string} piecePath
   * @param {number} offset
   */
  const resolveSlot = (piecePath, offset) => {
    const file = fileLayout.get(piecePath);
    if (!file) return -1;
    const localIdx = findPieceIndex(file.pieces, offset);
    if (localIdx < 0) return -1;
    return (file.baseIndex + localIdx) * 4;
  };

  const flushDirty = () => {
    if (!dirty || savedDirtyGen === dirtyGen) return flushChain;
    const targetGen = dirtyGen;
    const snapshot = new Uint8Array(tableBuffer).slice();
    flushChain = flushChain.then(async () => {
      await writeVerifiedPayloadAsync(stateFilePath, snapshot);
      savedDirtyGen = targetGen;
    }).catch(() => {});
    return flushChain;
  };

  const scheduleFlush = () => {
    if (closed || flushTimer !== null) return;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      void flushDirty();
    }, 50);
    flushTimer.unref?.();
  };

  return Object.freeze({
    directory,
    /**
     * @param {string} piecePath
     * @param {number} offset
     * @param {import('node:fs').BigIntStats} stat
     */
    isVerified(piecePath, offset, stat) {
      const slot = resolveSlot(piecePath, offset);
      if (slot < 0) return false;
      const ino = table[slot + 1];
      if (ino === 0n) return false;
      return table[slot] === BigInt.asUintN(64, stat.dev)
        && ino === BigInt.asUintN(64, stat.ino)
        && table[slot + 2] === BigInt.asUintN(64, stat.mtimeNs)
        && table[slot + 3] === BigInt.asUintN(64, stat.ctimeNs);
    },
    /**
     * @param {string} piecePath
     * @param {number} offset
     * @param {import('node:fs').BigIntStats} stat
     */
    markVerified(piecePath, offset, stat) {
      const slot = resolveSlot(piecePath, offset);
      if (slot < 0) return;
      table[slot] = BigInt.asUintN(64, stat.dev);
      table[slot + 1] = BigInt.asUintN(64, stat.ino);
      table[slot + 2] = BigInt.asUintN(64, stat.mtimeNs);
      table[slot + 3] = BigInt.asUintN(64, stat.ctimeNs);
      dirty = true;
      dirtyGen++;
      scheduleFlush();
    },
    async close() {
      closed = true;
      if (flushTimer !== null) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      await flushDirty();
    },
  });
}
