import { computeHash, createStreamingHasher } from '../../storage/shard-manager.js';
import { toArrayBuffer } from '../../formats/array-buffer.js';
import { encodeUtf8 } from '../../formats/encode-utf8.js';
import {
  getSourceRuntimeMetadata,
  normalizeHashString,
  normalizePositiveInteger,
} from './bundle-plan.js';

const SOURCE_VERIFY_CHUNK_BYTES = 4 * 1024 * 1024;

function toUint8Chunk(value, label) {
  return value instanceof Uint8Array ? value : new Uint8Array(toArrayBuffer(value, label));
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function resolveSourceEntry(index, manifest, shardSources) {
  const shard = manifest?.shards?.[index];
  if (!shard) {
    throw new Error(`Source shard index out of bounds: ${index}`);
  }
  const source = shardSources[index];
  if (!source) {
    throw new Error(`Missing source shard entry for index ${index}`);
  }
  return {
    sourcePath: source.path,
    shardSize: Number.isFinite(source.size) ? source.size : shard.size,
  };
}

export function createSourceStorageContext(options = {}) {
  const model = options.model ?? options.manifest;
  if (!model || typeof model !== 'object') {
    throw new Error('source storage context: model is required.');
  }

  const sourceRuntime = getSourceRuntimeMetadata(model);
  const shardSources = Array.isArray(options.shardSources) && options.shardSources.length > 0
    ? options.shardSources
    : (sourceRuntime?.sourceFiles ?? null);
  if (!shardSources || shardSources.length === 0) {
    throw new Error('source storage context: shardSources[] is required.');
  }

  const readRange = options.readRange;
  if (typeof readRange !== 'function') {
    throw new Error('source storage context: readRange(path, offset, length) is required.');
  }

  const streamRange = typeof options.streamRange === 'function'
    ? options.streamRange
    : null;
  const readText = typeof options.readText === 'function'
    ? options.readText
    : null;
  const readBinary = typeof options.readBinary === 'function'
    ? options.readBinary
    : null;
  const close = typeof options.close === 'function'
    ? options.close
    : null;
  const sourceFileMap = new Map(
    shardSources.map((entry) => [entry.path, entry])
  );
  const auxiliaryFileMap = new Map(
    (sourceRuntime?.auxiliaryFiles ?? []).map((entry) => [entry.path, entry])
  );
  const tokenizerJsonPath = options.tokenizerJsonPath ?? sourceRuntime?.tokenizer?.jsonPath ?? null;
  const tokenizerModelPath = options.tokenizerModelPath ?? sourceRuntime?.tokenizer?.modelPath ?? null;
  const verifyHashes = options.verifyHashes === true;
  const sourceHashesTrusted = options.sourceHashesTrusted === true;
  const verifiedSourceTasks = new Map();

  async function ensureVerifiedSource(sourcePath) {
    if (!verifyHashes || sourceHashesTrusted) {
      return;
    }
    let task = verifiedSourceTasks.get(sourcePath);
    if (!task) {
      task = (async () => {
        const descriptor = sourceFileMap.get(sourcePath);
        if (!descriptor) {
          throw new Error(`Missing source descriptor for ${sourcePath}.`);
        }
        const expectedHash = normalizeHashString(descriptor.hash, `source file hash (${sourcePath})`);
        if (!expectedHash) {
          throw new Error(
            `Source file "${sourcePath}" is missing a hash digest. ` +
            'Persist a materialized direct-source manifest or rebuild the synthetic bundle.'
          );
        }
        const hasher = await createStreamingHasher(descriptor.hashAlgorithm);
        const totalBytes = normalizePositiveInteger(descriptor.size, `source file size (${sourcePath})`);
        if (streamRange) {
          for await (const chunk of streamRange(
            sourcePath,
            0,
            totalBytes,
            { chunkBytes: SOURCE_VERIFY_CHUNK_BYTES }
          )) {
            hasher.update(toUint8Chunk(chunk, `streamRange(${sourcePath})`));
          }
        } else {
          let produced = 0;
          while (produced < totalBytes) {
            const nextLength = Math.min(SOURCE_VERIFY_CHUNK_BYTES, totalBytes - produced);
            const payload = await readRange(sourcePath, produced, nextLength);
            const bytes = toUint8Chunk(payload, `readRange(${sourcePath})`);
            if (bytes.byteLength <= 0) {
              break;
            }
            produced += bytes.byteLength;
            hasher.update(bytes);
          }
          if (produced !== totalBytes) {
            throw new Error(
              `Source file short read for verification (${sourcePath}): ` +
              `expected=${totalBytes}, got=${produced}.`
            );
          }
        }
        const computedHash = bytesToHex(await hasher.finalize());
        if (computedHash !== expectedHash) {
          throw new Error(
            `Source file hash mismatch for ${sourcePath}. ` +
            `Expected ${expectedHash}, got ${computedHash}.`
          );
        }
      })();
      verifiedSourceTasks.set(sourcePath, task);
      task.catch(() => {
        if (verifiedSourceTasks.get(sourcePath) === task) {
          verifiedSourceTasks.delete(sourcePath);
        }
      });
    }
    await task;
  }

  const loadShardRange = async (index, offset = 0, length = null) => {
    const { sourcePath, shardSize } = resolveSourceEntry(index, model, shardSources);
    const start = normalizePositiveInteger(offset, `shard offset (${index})`);
    const maxLength = Math.max(0, shardSize - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `shard length (${index})`));
    if (requested <= 0) {
      return new ArrayBuffer(0);
    }
    await ensureVerifiedSource(sourcePath);
    const payload = await readRange(sourcePath, start, requested);
    return toArrayBuffer(payload, `readRange(${sourcePath})`);
  };

  const loadShard = async (index) => {
    const { shardSize } = resolveSourceEntry(index, model, shardSources);
    return loadShardRange(index, 0, shardSize);
  };

  const streamShardRange = async function* (index, offset = 0, length = null, streamOptions = {}) {
    const { sourcePath, shardSize } = resolveSourceEntry(index, model, shardSources);
    const start = normalizePositiveInteger(offset, `shard stream offset (${index})`);
    const maxLength = Math.max(0, shardSize - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `shard stream length (${index})`));
    if (requested <= 0) {
      return;
    }
    await ensureVerifiedSource(sourcePath);

    if (streamRange) {
      for await (const chunk of streamRange(sourcePath, start, requested, streamOptions)) {
        yield toUint8Chunk(chunk, `streamRange(${sourcePath})`);
      }
      return;
    }

    const chunkBytesRaw = Number(streamOptions?.chunkBytes);
    const chunkBytes = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
      ? Math.floor(chunkBytesRaw)
      : SOURCE_VERIFY_CHUNK_BYTES;
    let produced = 0;
    while (produced < requested) {
      const nextLength = Math.min(chunkBytes, requested - produced);
      const payload = await readRange(sourcePath, start + produced, nextLength);
      const bytes = toUint8Chunk(payload, `readRange(${sourcePath})`);
      if (bytes.byteLength <= 0) {
        break;
      }
      produced += bytes.byteLength;
      yield bytes;
      if (bytes.byteLength < nextLength) {
        break;
      }
    }
  };

  const loadTokenizerJson = readText && tokenizerJsonPath
    ? async () => {
      const raw = await readText(tokenizerJsonPath);
      if (typeof raw === 'string') {
        if (verifyHashes) {
          const descriptor = auxiliaryFileMap.get(tokenizerJsonPath);
          if (descriptor?.hash) {
            const computedHash = await computeHash(encodeUtf8(raw), descriptor.hashAlgorithm);
            if (computedHash !== descriptor.hash) {
              throw new Error(
                `Tokenizer asset hash mismatch for ${tokenizerJsonPath}. ` +
                `Expected ${descriptor.hash}, got ${computedHash}.`
              );
            }
          }
        }
        return JSON.parse(raw);
      }
      if (verifyHashes && raw && typeof raw === 'object') {
        throw new Error(
          `readText(${tokenizerJsonPath}) must return the original JSON string when verifyHashes=true.`
        );
      }
      if (raw && typeof raw === 'object') {
        return raw;
      }
      throw new Error(`readText(${tokenizerJsonPath}) did not return tokenizer JSON data.`);
    }
    : null;

  const loadTokenizerModel = readBinary
    ? async (pathHint) => {
      const targetPath = typeof pathHint === 'string' && pathHint.trim()
        ? pathHint
        : tokenizerModelPath;
      if (!targetPath) {
        return null;
      }
      const raw = await readBinary(targetPath);
      const buffer = toArrayBuffer(raw, `readBinary(${targetPath})`);
      if (verifyHashes) {
        const descriptor = auxiliaryFileMap.get(targetPath);
        if (descriptor?.hash) {
          const computedHash = await computeHash(new Uint8Array(buffer), descriptor.hashAlgorithm);
          if (computedHash !== descriptor.hash) {
            throw new Error(
              `Binary asset hash mismatch for ${targetPath}. Expected ${descriptor.hash}, got ${computedHash}.`
            );
          }
        }
      }
      if (buffer.byteLength <= 0) {
        throw new Error(`readBinary(${targetPath}) returned an empty tokenizer model payload.`);
      }
      return buffer;
    }
    : null;

  const loadAuxiliaryFile = readBinary
    ? async (targetPath) => {
      if (typeof targetPath !== 'string' || !targetPath.trim()) {
        throw new Error('loadAuxiliaryFile(path) requires a non-empty path.');
      }
      const raw = await readBinary(targetPath);
      const buffer = toArrayBuffer(raw, `readBinary(${targetPath})`);
      if (verifyHashes) {
        const descriptor = auxiliaryFileMap.get(targetPath);
        if (descriptor?.hash) {
          const computedHash = await computeHash(new Uint8Array(buffer), descriptor.hashAlgorithm);
          if (computedHash !== descriptor.hash) {
            throw new Error(
              `Auxiliary asset hash mismatch for ${targetPath}. Expected ${descriptor.hash}, got ${computedHash}.`
            );
          }
        }
      }
      if (buffer.byteLength <= 0) {
        throw new Error(`readBinary(${targetPath}) returned an empty auxiliary payload.`);
      }
      return buffer;
    }
    : null;

  return {
    loadShard,
    loadShardRange,
    streamShardRange,
    loadTokenizerJson,
    loadTokenizerModel,
    loadAuxiliaryFile,
    verifyHashes,
    close,
  };
}
