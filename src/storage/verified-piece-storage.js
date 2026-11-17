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
    ...(options.nodeGather.fileIdentities ? { fileIdentities: structuredClone(options.nodeGather.fileIdentities) } : {}) } : null;
  const reader = await createVerifiedPieceReader(pinned);
  /** @type {Awaited<ReturnType<import('./verified-piece-gather.js').createVerifiedPieceGather>> | null} */
  let gather = null;
  const { manifest, readRange, readInto, readBinary } = reader;
  try {
    pinned.signal?.throwIfAborted();
    if (nodeGather) {
      const { createVerifiedPieceGather } = await import('./verified-piece-gather.js');
      gather = await createVerifiedPieceGather({ ...nodeGather, manifestBytes, indexBytes,
        indexIdentity: pinned.indexIdentity, startupSignal: pinned.signal });
    }
    pinned.signal?.throwIfAborted();
    let closed = false;
    /** @type {Set<Promise<import('./verified-piece-gather.js').VerifiedRangeLease>>} */
    const operations = new Set();
    const close = async () => {
      closed = true;
      const results = await Promise.allSettled([reader.close(), gather?.close()]);
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
      // The approved index binds canonical shards to the manifest. Every returned
      // byte still passes owned-copy piece verification; whole-source scans would
      // fetch unrelated partitions. Neither flag bypasses the authenticated reader.
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
    return { manifest, storage, getReceipt: reader.getReceipt, getGatherReceipt: gather ? gather.getReceipt : null };
  } catch (error) {
    const outcomes = await Promise.allSettled([reader.close(), gather?.close()]);
    const failures = outcomes.flatMap(outcome => outcome.status === 'rejected' ? [outcome.reason] : []);
    if (failures.length) throw new AggregateError([error, ...failures], 'Verified source opening and cleanup failed.');
    throw error;
  }
}
