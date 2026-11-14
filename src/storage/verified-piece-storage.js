import { ARTIFACT_FORMAT_DIRECT_SOURCE, buildRDRRShardSources, createArtifactStorageContext, getArtifactFormat } from './artifact-storage-context.js';
import { createVerifiedPieceReader } from './verified-piece-reader.js';
import { getSourceRuntimeMetadata } from '../formats/source-runtime.js';

/** Artifact host adapter over the shared authenticated piece reader.
 * @type {import('./verified-piece-storage.js').createVerifiedPieceStorage} */
export async function createVerifiedPieceStorage(options) {
  const reader = await createVerifiedPieceReader(options);
  const { manifest, readRange, readBinary } = reader;
  try {
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
    const storage = createArtifactStorageContext({ manifest, readRange, readBinary, shardSources,
      tokenizerJsonPath: directSource ? manifest.tokenizer?.file : undefined,
      readText: async path => new TextDecoder().decode(await readBinary(path)),
      // The approved index binds canonical shards to the manifest. Every returned
      // byte still passes owned-copy piece verification; whole-source scans would
      // fetch unrelated partitions. Neither flag bypasses the authenticated reader.
      verifyHashes: true, hashesTrusted: true, sourceHashesTrusted: directSource, close: reader.close });
    storage.preflight = reader.preflight;
    return { manifest, storage, getReceipt: reader.getReceipt };
  } catch (error) {
    await reader.close();
    throw error;
  }
}
