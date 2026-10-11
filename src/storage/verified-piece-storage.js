import { createArtifactStorageContext } from './artifact-storage-context.js';
import { createVerifiedPieceReader } from './verified-piece-reader.js';

/** RDRR host adapter over the shared authenticated piece reader.
 * @type {import('./verified-piece-storage.js').createVerifiedPieceStorage} */
export async function createVerifiedPieceStorage(options) {
  const reader = await createVerifiedPieceReader(options);
  const { manifest, readRange, readBinary } = reader;
  const storage = createArtifactStorageContext({ manifest, readRange, readBinary,
    readText: async path => new TextDecoder().decode(await readBinary(path)),
    // The shared reader verifies pinned pieces; whole-shard verification would
    // acquire unrelated partitions. The producer verifies whole source shards.
    verifyHashes: true, hashesTrusted: true, close: reader.close });
  storage.preflight = reader.preflight;
  return { manifest, storage, getReceipt: reader.getReceipt };
}
