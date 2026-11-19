import { ARTIFACT_FORMAT_DIRECT_SOURCE, buildRDRRShardSources, createArtifactStorageContext, getArtifactFormat } from './artifact-storage-context.js';
import { createVerifiedPieceReader } from './verified-piece-reader.js';
import { getSourceRuntimeMetadata } from '../formats/source-runtime.js';

/** Artifact host adapter over the shared authenticated piece reader.
 * @type {import('./verified-piece-storage.js').createVerifiedPieceStorage} */
export async function createVerifiedPieceStorage(options) {
  options.signal?.throwIfAborted();
  const manifestBytes = new Uint8Array(options.manifestBytes instanceof Uint8Array ? options.manifestBytes : new Uint8Array(options.manifestBytes));
  const indexBytes = new Uint8Array(options.indexBytes instanceof Uint8Array ? options.indexBytes : new Uint8Array(options.indexBytes));
  const pinned = { ...options, manifestBytes, indexBytes };
  const nodeGather = options.nodeGather ? { ...options.nodeGather, files: { ...options.nodeGather.files },
    ...(options.nodeGather.fileIdentities ? { fileIdentities: structuredClone(options.nodeGather.fileIdentities) } : {}),
    ...(options.nodeGather.protectedCache ? { protectedCache: structuredClone(options.nodeGather.protectedCache) } : {}),
    ...(options.nodeGather.verificationCache ? { verificationCache: structuredClone(options.nodeGather.verificationCache) } : {}) } : null;
  const nodePieceDirectory = options.nodePieceDirectory ? structuredClone(options.nodePieceDirectory) : null;
  if (nodePieceDirectory !== null && (pinned.acquire !== undefined || nodeGather !== null)) {
    throw new Error('Owned nodePieceDirectory cannot be combined with a generic acquire callback or nodeGather.');
  }
  if (nodePieceDirectory === null && typeof pinned.acquire !== 'function') {
    throw new Error('Verified piece storage requires an acquire callback or nodePieceDirectory.');
  }
  /** @type {Awaited<ReturnType<typeof import('./verified-piece-state-cache.js').prepareVerifiedPieceStateCache>>} */
  let sharedStateCache;
  /** @type {Awaited<ReturnType<typeof import('./verified-piece-state-cache.js').openPieceDirectoryStateCache>> | null} */
  let pieceDirStateCache = null;
  /** @type {import('./verified-piece-reader.js').VerifiedPieceReaderOptions['acquire']} */
  let readerAcquire;
  // Generic caller callbacks never inherit or seed file-bound verification state.
  let allowStateVerifiedAcquire = false;
  /** @type {(() => Promise<void>) | null} */
  let closePieceDirectory = null;
  /** @type {(() => import('./verified-piece-storage.js').NodePieceDirectoryReceipt) | undefined} */
  let getPieceDirectoryReceipt;

  if (nodeGather?.verificationCache !== undefined) {
    const parsedIndex = JSON.parse(new TextDecoder().decode(indexBytes));
    const { prepareVerifiedPieceStateCache } = await import('./verified-piece-state-cache.js');
    sharedStateCache = await prepareVerifiedPieceStateCache({
      rawPolicy: nodeGather.verificationCache,
      manifestIdentity: parsedIndex.manifestIdentity,
      indexIdentity: pinned.indexIdentity,
      index: parsedIndex,
      files: nodeGather.files,
    });
  }

  if (nodePieceDirectory !== null) {
    const parsedIndex = JSON.parse(new TextDecoder().decode(indexBytes));
    const [{ openPieceDirectoryStateCache }, fsPromises, fs, path] = await Promise.all([
      import('./verified-piece-state-cache.js'),
      import('node:fs/promises'),
      import('node:fs'),
      import('node:path'),
    ]);
    const { constants } = fs;
    if (typeof nodePieceDirectory.directory !== 'string' || !path.isAbsolute(nodePieceDirectory.directory)
      || !Number.isSafeInteger(nodePieceDirectory.maxPieceBytes) || nodePieceDirectory.maxPieceBytes < parsedIndex.maxPieceBytes
      || !Number.isSafeInteger(nodePieceDirectory.maxActiveReadBytes) || nodePieceDirectory.maxActiveReadBytes < nodePieceDirectory.maxPieceBytes) {
      throw new Error('Owned nodePieceDirectory requires an absolute directory and explicit piece/active-read bounds.');
    }
    const canonicalDir = await fsPromises.realpath(nodePieceDirectory.directory);
    const directoryStat = await fsPromises.stat(canonicalDir, { bigint: true });
    if (!directoryStat.isDirectory()) throw new Error('Owned nodePieceDirectory must be a directory.');
    if (nodePieceDirectory.verificationCache !== undefined) {
      if (!nodePieceDirectory.verificationCache || typeof nodePieceDirectory.verificationCache !== 'object'
        || Array.isArray(nodePieceDirectory.verificationCache)
        || typeof nodePieceDirectory.verificationCache.directory !== 'string'
        || !path.isAbsolute(nodePieceDirectory.verificationCache.directory)) {
        throw new Error('Owned nodePieceDirectory verificationCache requires an explicit absolute directory.');
      }
      pieceDirStateCache = await openPieceDirectoryStateCache({
        directory: nodePieceDirectory.verificationCache.directory,
        pieceDirectory: canonicalDir,
        manifestIdentity: parsedIndex.manifestIdentity,
        indexIdentity: pinned.indexIdentity,
        index: parsedIndex,
      });
    }
    if (!Number.isInteger(constants.O_NOFOLLOW) || !Number.isInteger(constants.O_NONBLOCK)) {
      throw new Error('Owned piece reads require O_NOFOLLOW and O_NONBLOCK support.');
    }
    const dirLifetime = new AbortController();
    /** @type {Set<Promise<unknown>>} */
    const dirOperations = new Set();
    /** Descriptors whose close failed; retried when the directory closes. @type {Set<number>} */
    const failedHandles = new Set();
    /** @param {number} fd @param {Uint8Array} target @param {number} offset @param {number} length @param {number} position
     * @returns {Promise<number>} */
    const readAt = (fd, target, offset, length, position) => new Promise((resolve, reject) => {
      fs.read(fd, target, offset, length, position, (error, bytesRead) => error ? reject(error) : resolve(bytesRead));
    });
    const dirStats = { reads: 0, activeReadBytes: 0, peakReadBytes: 0, missingPieces: 0 };
    let dirClosed = false;
    const stateCacheRef = pieceDirStateCache;
    allowStateVerifiedAcquire = stateCacheRef !== null;
    readerAcquire = (piece, control) => {
      const op = (async () => {
        const active = AbortSignal.any([
          dirLifetime.signal,
          ...(pinned.signal ? [pinned.signal] : []),
          ...(control?.signal ? [control.signal] : []),
        ]);
        active.throwIfAborted();
        if (dirClosed) throw new Error('Diagnostic piece cache closed');
        if (!/^[a-zA-Z0-9_.-]+$/u.test(piece.path) || ['.', '..'].includes(piece.path)
          || !Number.isSafeInteger(piece.offset) || piece.offset < 0
          || !Number.isSafeInteger(piece.size) || piece.size <= 0 || piece.size > nodePieceDirectory.maxPieceBytes) {
          throw new Error('Invalid authenticated cache piece');
        }
        if (dirStats.activeReadBytes + piece.size > nodePieceDirectory.maxActiveReadBytes) {
          throw new Error('Piece cache active-read budget exceeded');
        }
        dirStats.activeReadBytes += piece.size;
        dirStats.peakReadBytes = Math.max(dirStats.peakReadBytes, dirStats.activeReadBytes);
        // The identity checks, open and close are metadata calls on files the OS already holds; they run
        // synchronously so each piece queues one pool task (its read) instead of seven. Queued behind
        // the others, first-read hashing was waiting for the pool rather than for the processor.
        /** @type {number | undefined} */
        let handle;
        try {
          const currentDirectory = fs.lstatSync(canonicalDir, { bigint: true });
          if (!currentDirectory.isDirectory() || currentDirectory.dev !== directoryStat.dev
            || currentDirectory.ino !== directoryStat.ino) {
            throw new Error('Piece cache directory changed');
          }
          const filename = path.join(canonicalDir, `${piece.path}.${piece.offset}`);
          handle = fs.openSync(filename, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
          const before = fs.fstatSync(handle, { bigint: true });
          const linkBefore = fs.lstatSync(filename, { bigint: true });
          if (!before.isFile() || !linkBefore.isFile() || before.size !== BigInt(piece.size)
            || linkBefore.dev !== before.dev || linkBefore.ino !== before.ino || linkBefore.size !== before.size
            || linkBefore.mtimeNs !== before.mtimeNs || linkBefore.ctimeNs !== before.ctimeNs) {
            throw new Error('Cached piece must be a complete regular file');
          }
          const bytes = new Uint8Array(piece.size);
          let received = 0;
          while (received < bytes.length) {
            active.throwIfAborted();
            const bytesRead = await readAt(handle, bytes, received, bytes.length - received, received);
            if (bytesRead <= 0) throw new Error('Cached piece read truncated');
            received += bytesRead;
          }
          const after = fs.fstatSync(handle, { bigint: true });
          const linkAfter = fs.lstatSync(filename, { bigint: true });
          const unchanged = after.isFile() && linkAfter.isFile()
            && after.dev === before.dev && after.ino === before.ino && after.size === before.size
            && after.mtimeNs === before.mtimeNs && after.ctimeNs === before.ctimeNs
            && linkAfter.dev === before.dev && linkAfter.ino === before.ino && linkAfter.size === before.size
            && linkAfter.mtimeNs === before.mtimeNs && linkAfter.ctimeNs === before.ctimeNs;
          if (!unchanged) throw new Error('Cached piece changed during read');
          active.throwIfAborted();
          dirStats.reads++;
          if (!stateCacheRef) return bytes;
          return {
            bytes,
            stateVerified: stateCacheRef.isVerified(piece.path, piece.offset, after),
            onAuthenticated: () => { stateCacheRef.markVerified(piece.path, piece.offset, after); },
          };
        } catch (error) {
          if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') dirStats.missingPieces++;
          throw error;
        } finally {
          try {
            if (handle !== undefined) {
              try { fs.closeSync(handle); }
              catch (error) { failedHandles.add(handle); dirClosed = true; dirLifetime.abort(error); throw error; }
            }
          } finally { dirStats.activeReadBytes -= piece.size; }
        }
      })();
      dirOperations.add(op);
      op.finally(() => dirOperations.delete(op)).catch(() => {});
      return op;
    };
    closePieceDirectory = async () => {
      dirClosed = true;
      dirLifetime.abort(new Error('Diagnostic piece cache closed'));
      await Promise.allSettled([...dirOperations]);
      const closeFailures = [];
      for (const handle of [...failedHandles]) {
        try { fs.closeSync(handle); failedHandles.delete(handle); }
        catch (error) { closeFailures.push(error); }
      }
      if (closeFailures.length) throw new AggregateError(closeFailures, 'Diagnostic piece cache cleanup failed');
    };
    getPieceDirectoryReceipt = () => ({ ...dirStats, pendingDescriptorCloses: failedHandles.size });
  } else {
    readerAcquire = /** @type {import('./verified-piece-reader.js').VerifiedPieceReaderOptions['acquire']} */ (pinned.acquire);
  }

  let reader;
  try {
    reader = await createVerifiedPieceReader({ ...pinned, acquire: readerAcquire, allowStateVerifiedAcquire });
  } catch (error) {
    await Promise.allSettled([closePieceDirectory?.(), sharedStateCache?.close(), pieceDirStateCache?.close()]);
    throw error;
  }
  /** @type {Awaited<ReturnType<import('./verified-piece-gather.js').createVerifiedPieceGather>> | null} */
  let gather = null;
  const { manifest, readRange, readInto, readBinary } = reader;
  try {
    pinned.signal?.throwIfAborted();
    if (nodeGather) {
      const { createVerifiedPieceGather } = await import('./verified-piece-gather.js');
      gather = await createVerifiedPieceGather({ ...nodeGather, sharedStateCache, manifestBytes, indexBytes,
        indexIdentity: pinned.indexIdentity, startupSignal: pinned.signal });
    }
    pinned.signal?.throwIfAborted();
    let closed = false;
    /** @type {Set<Promise<import('./verified-piece-gather.js').VerifiedRangeLease>>} */
    const operations = new Set();
    const close = async () => {
      closed = true;
      const results = await Promise.allSettled([
        reader.close(),
        gather?.close(),
        closePieceDirectory?.(),
        sharedStateCache?.close(),
        pieceDirStateCache?.close(),
      ]);
      await Promise.allSettled([...operations]);
      const failed = results.find(result => result.status === 'rejected');
      if (failed) throw failed.reason;
    };
    const directSource = getArtifactFormat(manifest) === ARTIFACT_FORMAT_DIRECT_SOURCE;
    const shardSources = directSource
      ? buildRDRRShardSources(manifest).map(shard => {
        if (typeof shard.hash !== 'string' || !/^[a-f0-9]{64}$/.test(shard.hash)) {
          throw new Error('Verified source shard requires a complete hash.');
        }
        return { ...shard, hash: shard.hash, filename: shard.path };
      })
      : undefined;
    if (shardSources) {
      const sources = getSourceRuntimeMetadata(manifest)?.sourceFiles;
      if (!Array.isArray(sources) || sources.length !== shardSources.length) {
        throw new Error('Verified source shard binding mismatch.');
      }
      for (const [index, shard] of shardSources.entries()) {
        const source = sources[index];
        if (!source || typeof source.path !== 'string' || !source.path.trim()
          || source.index !== index || source.filename !== shard.path || source.size !== shard.size
          || source.hash !== shard.hash || (source.hashAlgorithm ?? manifest.hashAlgorithm) !== shard.hashAlgorithm) {
          throw new Error('Verified source shard binding mismatch.');
        }
      }
    }
    const storage = createArtifactStorageContext({ manifest, readRange, readInto, readBinary, shardSources,
      tokenizerJsonPath: directSource ? manifest.tokenizer?.file : undefined,
      readText: async path => new TextDecoder().decode(await readBinary(path)),
      verifyHashes: true, hashesTrusted: true, sourceHashesTrusted: directSource, close });
    storage.preflight = reader.preflight;
    if (gather) {
      const port = gather.rangeLease;
      /** @param {(signal: AbortSignal) => Promise<import('./verified-piece-gather.js').VerifiedRangeLease>} open
       * @param {AbortSignal} signal */
      const readScoped = (open, signal) => {
        const activeSignal = pinned.signal && signal ? AbortSignal.any([pinned.signal, signal]) : pinned.signal ?? signal;
        const operation = (async () => {
          activeSignal?.throwIfAborted();
          if (closed) throw new Error('Verified source storage closed.');
          const lease = await open(activeSignal);
          try {
            activeSignal?.throwIfAborted();
            if (closed) throw new Error('Verified source storage closed.');
            return lease;
          } catch (error) { lease.release(); throw error; }
        })();
        operations.add(operation);
        operation.finally(() => operations.delete(operation)).catch(() => {});
        return operation;
      };
      const { readMany, ...basePort } = port;
      /** @type {import('./verified-piece-gather.js').VerifiedRangeLeasePort['readMany']} */
      const scopedReadMany = typeof readMany === 'function'
        ? (ranges, signal) => readScoped(active => readMany.call(port, ranges, active), signal) : undefined;
      /** @type {import('./verified-piece-gather.js').VerifiedRangeLeasePort} */
      const scopedPort = { ...basePort,
        read: (shard, offset, length, signal) => readScoped(active => port.read(shard, offset, length, active), signal),
        ...(scopedReadMany ? { readMany: scopedReadMany } : {}),
      };
      storage.rangeLease = Object.freeze(scopedPort);
    }
    return {
      manifest,
      storage,
      getReceipt: reader.getReceipt,
      getGatherReceipt: gather ? gather.getReceipt : null,
      ...(getPieceDirectoryReceipt ? { getPieceDirectoryReceipt } : {}),
    };
  } catch (error) {
    const outcomes = await Promise.allSettled([
      reader.close(),
      gather?.close(),
      closePieceDirectory?.(),
      sharedStateCache?.close(),
      pieceDirStateCache?.close(),
    ]);
    const failures = outcomes.flatMap(outcome => outcome.status === 'rejected' ? [outcome.reason] : []);
    if (failures.length) throw new AggregateError([error, ...failures], 'Verified source opening and cleanup failed.');
    throw error;
  }
}
