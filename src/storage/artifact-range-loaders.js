import { normalizePositiveInteger } from '../formats/source-runtime.js';
import { toArrayBuffer } from '../formats/array-buffer.js';

/** Exact range contracts for path-backed RDRR shards.
 * @type {import('./artifact-range-loaders.js').createArtifactRangeLoaders} */
export function createArtifactRangeLoaders({ shardSources, readRange, readInto, ensureVerifiedSource }) {
  /** @type {NonNullable<import('./artifact-storage-context.js').ArtifactStorageContext['loadShardRange']>} */
  const loadShardRange = async (index, offset = 0, length = null) => {
    const descriptor = shardSources[index];
    if (!descriptor) {
      throw new Error(`Artifact shard index out of bounds: ${index}`);
    }
    const start = normalizePositiveInteger(offset, `artifact shard offset (${index})`);
    const maxLength = Math.max(0, descriptor.size - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `artifact shard length (${index})`));
    if (requested <= 0) {
      return new ArrayBuffer(0);
    }
    await ensureVerifiedSource(descriptor.path);
    const payload = await readRange(descriptor.path, start, requested);
    return toArrayBuffer(payload, `readRange(${descriptor.path})`);
  };

  /** @type {import('./artifact-storage-context.js').ArtifactStorageContext['loadShardRangeInto']} */
  const loadShardRangeInto = readInto ? async (index, offset, destination) => {
    const descriptor = shardSources[index];
    if (!descriptor) throw new Error(`Artifact shard index out of bounds: ${index}`);
    const start = normalizePositiveInteger(offset, `artifact shard offset (${index})`);
    if (!(destination instanceof Uint8Array) || !Number.isSafeInteger(start + destination.byteLength)
      || start + destination.byteLength > descriptor.size) {
      throw new Error('Artifact readInto destination exceeds the declared shard range.');
    }
    await ensureVerifiedSource(descriptor.path);
    const count = await readInto(descriptor.path, start, destination);
    if (count !== destination.byteLength) throw new Error(`Artifact readInto short read: ${count} of ${destination.byteLength}.`);
    return count;
  } : null;

  return { loadShardRange, loadShardRangeInto };
}
