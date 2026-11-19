/*
 * Node-only direct reads of ranges whose every piece is already verified for the file's current
 * state. The bytes go from the approved file into caller-owned memory on the I/O pool, with no
 * worker hop and no intermediate copy; the caller may pass a mapped GPU staging range.
 *
 * It applies the checks a gather lane applies to a state-verified piece: the open descriptor and
 * the approved pathname must both still be the admitted file (device, inode, size) in the exact
 * state its pieces were verified in (also mtime and ctime), before the read and again after it.
 * A range with any unverified piece, or a file that no longer matches, is declined: first-time
 * verification stays with the lanes, which hash before they publish.
 */
import { openSync, closeSync, fstatSync, lstatSync, read as readAsync, constants } from 'node:fs';

/** @type {import('./verified-piece-direct.js').createVerifiedDirectReader} */
export function createVerifiedDirectReader({ files, fileIdentities, index, stateCache }) {
  if (!Number.isInteger(constants.O_NOFOLLOW) || !Number.isInteger(constants.O_NONBLOCK)) {
    throw new Error('Direct verified reads require O_NOFOLLOW and O_NONBLOCK support.');
  }
  const approved = { ...files };
  const identities = structuredClone(fileIdentities);
  /** @type {Map<string, ReadonlyArray<{ offset: number, size: number }>>} */
  const pieces = new Map(index.files.map(file => [file.path, file.pieces]));
  /** @type {Map<string, number>} */
  const handles = new Map();
  /** @type {Set<Promise<boolean>>} */
  const operations = new Set();
  const stats = { reads: 0, bytes: 0, declined: 0 };
  let closed = false;
  /** @type {Promise<void> | null} */
  let closing = null;

  /** @param {string} path @param {import('node:fs').BigIntStats} stat */
  const admitted = (path, stat) => {
    const expected = identities[path];
    return !!expected && stat.isFile() && stat.dev.toString() === expected.device
      && stat.ino.toString() === expected.inode && stat.size.toString() === expected.size;
  };
  /** @param {string} path @param {number} handle */
  const current = (path, handle) => {
    const descriptor = fstatSync(handle, { bigint: true });
    const pathname = lstatSync(approved[path], { bigint: true });
    return admitted(path, descriptor) && admitted(path, pathname)
      && stateCache.checkFileStat(path, descriptor) && stateCache.checkFileStat(path, pathname);
  };
  /** Every byte of the range lies in a piece verified for the current file state.
   * @param {string} path @param {number} offset @param {number} length */
  const verified = (path, offset, length) => {
    const list = pieces.get(path);
    if (!list) return false;
    let low = 0, high = list.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (list[mid].offset + list[mid].size <= offset) low = mid + 1; else high = mid;
    }
    let end = offset;
    for (let at = low; at < list.length && end < offset + length; at++) {
      const piece = list[at];
      if (piece.offset > end || !stateCache.isPieceVerified(path, piece.offset)) return false;
      end = piece.offset + piece.size;
    }
    return end >= offset + length;
  };
  /** @param {number} handle @param {Uint8Array} target @param {number} at @param {number} length @param {number} position
   * @returns {Promise<number>} */
  const readAt = (handle, target, at, length, position) => new Promise((resolve, reject) => {
    readAsync(handle, target, at, length, position, (error, bytesRead) => error ? reject(error) : resolve(bytesRead));
  });

  /** @param {string} path @param {number} offset @param {Uint8Array} destination @param {AbortSignal | undefined} signal */
  async function readOwned(path, offset, destination, signal) {
    signal?.throwIfAborted();
    if (closed) throw new Error('Gather reader closed.');
    if (!(destination instanceof Uint8Array) || destination.byteLength < 1
      || !Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(offset + destination.byteLength)) {
      throw new Error('Direct verified read requires a non-empty destination and a safe range.');
    }
    const length = destination.byteLength;
    if (!Object.hasOwn(approved, path) || !verified(path, offset, length)) { stats.declined++; return false; }
    let handle = handles.get(path);
    if (handle === undefined) {
      handle = openSync(approved[path], constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
      handles.set(path, handle);
    }
    if (!current(path, handle)) { stats.declined++; return false; }
    let received = 0;
    while (received < length) {
      const count = await readAt(handle, destination, received, length - received, offset + received);
      if (count <= 0) throw new Error('Direct verified read truncated.');
      received += count;
    }
    // The destination holds authentic bytes only if the file stayed the verified one throughout.
    if (!current(path, handle) || !verified(path, offset, length)) {
      throw new Error('Gather source changed during a direct verified read.');
    }
    if (closed) throw new Error('Gather reader closed.');
    signal?.throwIfAborted();
    stats.reads++; stats.bytes += length;
    return true;
  }

  /** @type {import('./verified-piece-direct.js').VerifiedDirectReader} */
  const reader = {
    readInto(path, offset, destination, signal) {
      const operation = readOwned(path, offset, destination, signal);
      operations.add(operation);
      operation.finally(() => operations.delete(operation)).catch(() => {});
      return operation;
    },
    getStats: () => ({ ...stats }),
    close() {
      if (closing) return closing;
      closed = true;
      closing = (async () => {
        // A descriptor is never closed under a read that still targets caller memory.
        await Promise.allSettled([...operations]);
        const failures = [];
        for (const [path, handle] of handles) {
          try { closeSync(handle); handles.delete(path); } catch (error) { failures.push(error); }
        }
        if (failures.length) throw new AggregateError(failures, 'Direct verified reader cleanup failed.');
      })();
      closing.catch(() => { closing = null; });
      return closing;
    },
  };
  return Object.freeze(reader);
}
