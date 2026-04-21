import {
  DIRECT_SOURCE_RUNTIME_MODE,
  createSourceStorageContext,
} from '../tooling/source-runtime-bundle.js';
import { createStreamingHasher } from './shard-manager.js';
import { isNodeRuntime } from '../utils/runtime-env.js';
import { toArrayBuffer } from '../utils/array-buffer.js';

export const ARTIFACT_FORMAT_RDRR = 'rdrr';
export const ARTIFACT_FORMAT_DIRECT_SOURCE = 'direct-source';

const SOURCE_VERIFY_CHUNK_BYTES = 4 * 1024 * 1024;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeArtifactPath(value) {
  return normalizeText(value).replace(/\\/g, '/').replace(/^\/+/, '');
}

function normalizeHashAlgorithm(value) {
  const normalized = normalizeText(value).toLowerCase();
  return normalized === 'blake3' ? 'blake3' : 'sha256';
}

function normalizeHashString(value, label) {
  if (value == null) {
    return null;
  }
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) {
    return null;
  }
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`${label} must be a 64-character lowercase hex digest.`);
  }
  return normalized;
}

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return Math.floor(parsed);
}

function normalizeOptionalPositiveInteger(value, fallback = 0) {
  if (value == null) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return fallback;
  return Math.floor(parsed);
}

function toUint8Chunk(value, label) {
  return value instanceof Uint8Array ? value : new Uint8Array(toArrayBuffer(value, label));
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function createHttpArtifactFetchError(error, url, offset = null, length = null) {
  const message = error?.message || String(error);
  const range = Number.isFinite(offset) && Number.isFinite(length)
    ? ` offset=${Math.floor(offset)} length=${Math.floor(length)}`
    : '';
  const wrapped = new Error(
    `HTTP artifact fetch failed for ${url}${range}: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    artifactUrl: url,
    offset: Number.isFinite(offset) ? Math.floor(offset) : null,
    length: Number.isFinite(length) ? Math.floor(length) : null,
  };
  return wrapped;
}

function getNodeFileCacheKey(fileRef) {
  if (typeof fileRef === 'string') {
    return fileRef;
  }
  if (fileRef instanceof URL) {
    return fileRef.href;
  }
  return String(fileRef);
}

function getSourceRuntimeMetadata(manifest) {
  const metadata = manifest?.metadata?.sourceRuntime;
  if (!metadata || typeof metadata !== 'object') {
    return null;
  }
  const mode = normalizeText(metadata.mode).toLowerCase();
  if (mode !== DIRECT_SOURCE_RUNTIME_MODE) {
    return null;
  }
  const sourceKind = normalizeText(metadata.sourceKind).toLowerCase() || null;
  return {
    mode,
    sourceKind,
  };
}

export function getArtifactFormat(manifest) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (sourceRuntime) {
    if (sourceRuntime.sourceKind === ARTIFACT_FORMAT_RDRR) {
      return ARTIFACT_FORMAT_RDRR;
    }
    return ARTIFACT_FORMAT_DIRECT_SOURCE;
  }
  if (Array.isArray(manifest?.shards) && manifest.shards.length > 0) {
    return ARTIFACT_FORMAT_RDRR;
  }
  return null;
}

function resolveManifestTokenizerPaths(manifest) {
  const tokenizer = manifest?.tokenizer;
  if (!tokenizer || typeof tokenizer !== 'object') {
    return {
      jsonPath: null,
      modelPath: null,
    };
  }
  const tokenizerType = normalizeText(tokenizer.type).toLowerCase();
  const bundledTokenizerPath = normalizeArtifactPath(tokenizer.file);
  const sentencepieceModelPath = normalizeArtifactPath(tokenizer.sentencepieceModel);
  return {
    jsonPath: (
      (tokenizerType === 'bundled' || tokenizerType === 'huggingface')
      && bundledTokenizerPath
    ) ? bundledTokenizerPath : null,
    modelPath: sentencepieceModelPath || (tokenizerType === 'sentencepiece' ? 'tokenizer.model' : null),
  };
}

function buildRDRRShardSources(manifest) {
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  if (shards.length === 0) {
    throw new Error(
      `Artifact "${manifest?.modelId ?? 'unknown'}" is missing shards; ` +
      're-convert or provide a valid manifest.'
    );
  }
  const manifestHashAlgorithm = normalizeHashAlgorithm(manifest?.hashAlgorithm);
  return shards.map((shard, index) => {
    const path = normalizeArtifactPath(shard?.filename);
    if (!path) {
      throw new Error(`Artifact shard ${index} is missing filename.`);
    }
    return {
      index,
      path,
      size: normalizePositiveInteger(shard?.size, `artifact shard size (${path})`),
      hash: normalizeHashString(shard?.hash, `artifact shard hash (${path})`),
      hashAlgorithm: normalizeHashAlgorithm(shard?.hashAlgorithm ?? manifestHashAlgorithm),
    };
  });
}

function createPathBackedStorageContext(options) {
  const shardSources = Array.isArray(options?.shardSources) ? options.shardSources : [];
  if (shardSources.length === 0) {
    throw new Error('artifact storage context: shardSources[] is required.');
  }

  const readRange = options?.readRange;
  if (typeof readRange !== 'function') {
    throw new Error('artifact storage context: readRange(path, offset, length) is required.');
  }

  const streamRange = typeof options?.streamRange === 'function'
    ? options.streamRange
    : null;
  const readText = typeof options?.readText === 'function'
    ? options.readText
    : null;
  const readBinary = typeof options?.readBinary === 'function'
    ? options.readBinary
    : null;
  const close = typeof options?.close === 'function'
    ? options.close
    : null;
  const tokenizerJsonPath = options?.tokenizerJsonPath ?? null;
  const tokenizerModelPath = options?.tokenizerModelPath ?? null;
  const tensorsJsonPath = options?.tensorsJsonPath ?? null;
  const verifyHashes = options?.verifyHashes === true;
  const hashesTrusted = options?.hashesTrusted === true;

  const descriptorByPath = new Map(shardSources.map((entry) => [entry.path, entry]));
  const verifiedSourceTasks = new Map();

  async function ensureVerifiedSource(sourcePath) {
    if (!verifyHashes || hashesTrusted) {
      return;
    }
    let task = verifiedSourceTasks.get(sourcePath);
    if (!task) {
      task = (async () => {
        const descriptor = descriptorByPath.get(sourcePath);
        if (!descriptor) {
          throw new Error(`Missing artifact shard descriptor for ${sourcePath}.`);
        }
        const expectedHash = normalizeHashString(
          descriptor.hash,
          `artifact shard hash (${sourcePath})`
        );
        if (!expectedHash) {
          throw new Error(`Artifact shard "${sourcePath}" is missing a hash digest.`);
        }
        const hasher = await createStreamingHasher(descriptor.hashAlgorithm);
        const totalBytes = normalizePositiveInteger(
          descriptor.size,
          `artifact shard size (${sourcePath})`
        );
        if (streamRange) {
          for await (const chunk of streamRange(sourcePath, 0, totalBytes, { chunkBytes: SOURCE_VERIFY_CHUNK_BYTES })) {
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
              `Artifact shard short read for verification (${sourcePath}): ` +
              `expected=${totalBytes}, got=${produced}.`
            );
          }
        }
        const computedHash = bytesToHex(await hasher.finalize());
        if (computedHash !== expectedHash) {
          throw new Error(
            `Artifact shard hash mismatch for ${sourcePath}. ` +
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

  const loadShard = async (index) => {
    const descriptor = shardSources[index];
    if (!descriptor) {
      throw new Error(`Artifact shard index out of bounds: ${index}`);
    }
    return loadShardRange(index, 0, descriptor.size);
  };

  const preflight = async () => {
    const failures = [];
    for (const descriptor of shardSources) {
      try {
        if (descriptor.size > 0) {
          const tail = await readRange(descriptor.path, descriptor.size - 1, 1);
          const bytes = toUint8Chunk(tail, `readRange(${descriptor.path})`);
          if (bytes.byteLength < 1) {
            failures.push(`${descriptor.path}: expected at least ${descriptor.size} bytes`);
          }
        } else {
          await readRange(descriptor.path, 0, 0);
        }
      } catch (error) {
        failures.push(`${descriptor.path}: ${error.message}`);
      }
    }
    if (readText && tokenizerJsonPath) {
      try {
        await readText(tokenizerJsonPath);
      } catch (error) {
        failures.push(`${tokenizerJsonPath}: ${error.message}`);
      }
    }
    if (readBinary && tokenizerModelPath) {
      try {
        await readBinary(tokenizerModelPath);
      } catch (error) {
        failures.push(`${tokenizerModelPath}: ${error.message}`);
      }
    }
    if (readText && tensorsJsonPath) {
      try {
        await readText(tensorsJsonPath);
      } catch (error) {
        failures.push(`${tensorsJsonPath}: ${error.message}`);
      }
    }
    if (failures.length > 0) {
      throw new Error(`Artifact contract preflight failed: ${failures.join('; ')}`);
    }
  };

  const streamShardRange = async function* (index, offset = 0, length = null, streamOptions = {}) {
    const descriptor = shardSources[index];
    if (!descriptor) {
      throw new Error(`Artifact shard index out of bounds: ${index}`);
    }
    const start = normalizePositiveInteger(offset, `artifact shard stream offset (${index})`);
    const maxLength = Math.max(0, descriptor.size - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `artifact shard stream length (${index})`));
    if (requested <= 0) {
      return;
    }
    await ensureVerifiedSource(descriptor.path);

    if (streamRange) {
      for await (const chunk of streamRange(descriptor.path, start, requested, streamOptions)) {
        yield toUint8Chunk(chunk, `streamRange(${descriptor.path})`);
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
      const payload = await readRange(descriptor.path, start + produced, nextLength);
      const bytes = toUint8Chunk(payload, `readRange(${descriptor.path})`);
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

  return {
    preflight,
    loadShard,
    loadShardRange,
    streamShardRange,
    loadTokenizerJson: readText && tokenizerJsonPath
      ? async () => {
        const payload = await readText(tokenizerJsonPath);
        if (payload == null) return null;
        if (typeof payload === 'string') {
          return JSON.parse(payload);
        }
        if (typeof payload === 'object') {
          return payload;
        }
        throw new Error('artifact storage context: tokenizer JSON must load as string or object.');
      }
      : null,
    loadTokenizerModel: readBinary && tokenizerModelPath
      ? async (pathHint = null) => {
        const selectedPath = normalizeArtifactPath(pathHint) || tokenizerModelPath;
        const payload = await readBinary(selectedPath);
        return payload == null ? null : toArrayBuffer(payload, `readBinary(${selectedPath})`);
      }
      : null,
    loadTensorsJson: readText && tensorsJsonPath
      ? async () => {
        const payload = await readText(tensorsJsonPath);
        return payload == null ? null : payload;
      }
      : null,
    verifyHashes,
    close,
  };
}

export function createArtifactStorageContext(options = {}) {
  const manifest = options?.manifest;
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('artifact storage context: manifest is required.');
  }

  const format = getArtifactFormat(manifest);
  if (!format) {
    throw new Error(
      `Unsupported artifact manifest for "${manifest?.modelId ?? 'unknown'}". ` +
      'Expected an RDRR or direct-source artifact manifest.'
    );
  }

  const expectedFormat = options?.expectedFormat ?? null;
  if (expectedFormat && format !== expectedFormat) {
    throw new Error(
      `artifact storage context: expected format "${expectedFormat}", got "${format}".`
    );
  }

  if (format === ARTIFACT_FORMAT_DIRECT_SOURCE) {
    return createSourceStorageContext(options);
  }

  const tokenizerPaths = resolveManifestTokenizerPaths(manifest);
  return createPathBackedStorageContext({
    shardSources: buildRDRRShardSources(manifest),
    readRange: options.readRange,
    streamRange: options.streamRange,
    readText: options.readText,
    readBinary: options.readBinary,
    close: options.close,
    tokenizerJsonPath: options.tokenizerJsonPath ?? tokenizerPaths.jsonPath,
    tokenizerModelPath: options.tokenizerModelPath ?? tokenizerPaths.modelPath,
    tensorsJsonPath: normalizeArtifactPath(manifest.tensorsFile),
    verifyHashes: options.verifyHashes === true,
    hashesTrusted: options.hashesTrusted === true,
  });
}

function normalizeBaseUrl(baseUrl) {
  if (typeof baseUrl !== 'string' || baseUrl.trim().length === 0) {
    return null;
  }
  return baseUrl.replace(/\/$/, '');
}

function isFileUrlBaseUrl(baseUrl) {
  if (typeof baseUrl !== 'string' || !baseUrl) {
    return false;
  }
  try {
    return new URL(baseUrl).protocol === 'file:';
  } catch {
    return false;
  }
}

function isNodeFilesystemPathBaseUrl(baseUrl) {
  if (typeof baseUrl !== 'string' || !baseUrl || isFileUrlBaseUrl(baseUrl)) {
    return false;
  }
  return baseUrl.startsWith('/') || /^[a-zA-Z]:[\\/]/.test(baseUrl);
}

async function resolveArtifactFileReference(root, relativePath) {
  const normalizedPath = normalizeArtifactPath(relativePath);
  if (!normalizedPath) {
    throw new Error('Artifact file path is required.');
  }
  if (isFileUrlBaseUrl(root)) {
    return new URL(normalizedPath, `${root}/`);
  }
  if (isNodeFilesystemPathBaseUrl(root)) {
    const path = await import('node:path');
    return path.join(root, normalizedPath);
  }
  throw new Error(`Unsupported local artifact root "${root}".`);
}

function createNodeFileAccess() {
  const readers = new Map();

  const getReader = (fileRef) => {
    const cacheKey = getNodeFileCacheKey(fileRef);
    let reader = readers.get(cacheKey);
    if (reader) {
      return reader;
    }
    let handlePromise = null;
    let sizePromise = null;
    let closed = false;
    const ensureHandle = async () => {
      if (closed) {
        throw new Error(`artifact storage context: file reader already closed for "${cacheKey}".`);
      }
      if (!handlePromise) {
        handlePromise = import('node:fs/promises')
          .then((fs) => fs.open(fileRef, 'r'))
          .catch((error) => {
            handlePromise = null;
            throw error;
          });
      }
      return handlePromise;
    };
    const getSize = async () => {
      if (!sizePromise) {
        sizePromise = (async () => {
          const handle = await ensureHandle();
          const stats = await handle.stat();
          return Number(stats.size);
        })().catch((error) => {
          sizePromise = null;
          throw error;
        });
      }
      return sizePromise;
    };
    reader = {
      async readRange(offset = 0, length = null) {
        const start = Math.max(0, Math.floor(Number(offset) || 0));
        const fileSize = await getSize();
        const end = length == null
          ? fileSize
          : Math.min(fileSize, start + Math.max(0, Math.floor(Number(length) || 0)));
        if (end <= start) {
          return new ArrayBuffer(0);
        }
        const handle = await ensureHandle();
        const out = Buffer.allocUnsafe(end - start);
        let position = 0;
        while (position < out.length) {
          const { bytesRead } = await handle.read(out, position, out.length - position, start + position);
          if (bytesRead === 0) {
            break;
          }
          position += bytesRead;
        }
        return out.buffer.slice(out.byteOffset, out.byteOffset + position);
      },
      async getSize() {
        return getSize();
      },
      async close() {
        closed = true;
        const handle = await handlePromise;
        handlePromise = null;
        if (handle) {
          await handle.close();
        }
      },
    };
    readers.set(cacheKey, reader);
    return reader;
  };

  return {
    async readRange(fileRef, offset = 0, length = null) {
      return getReader(fileRef).readRange(offset, length);
    },
    async getSize(fileRef) {
      return getReader(fileRef).getSize();
    },
    async close() {
      const pending = Array.from(readers.values(), (reader) => reader.close());
      readers.clear();
      await Promise.all(pending);
    },
  };
}

export function createNodeFileArtifactStorageContext(baseUrl, manifest) {
  const root = normalizeBaseUrl(baseUrl);
  if (
    !root
    || !manifest
    || !isNodeRuntime()
    || (!isFileUrlBaseUrl(root) && !isNodeFilesystemPathBaseUrl(root))
  ) {
    return null;
  }

  const fileAccess = createNodeFileAccess();

  try {
    const readRange = async (relativePath, offset = 0, length = null) => {
      const fileRef = await resolveArtifactFileReference(root, relativePath);
      return fileAccess.readRange(fileRef, offset, length);
    };
    const streamRange = async function* (relativePath, offset = 0, length = null, options = {}) {
      if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
        return;
      }
      const fileRef = await resolveArtifactFileReference(root, relativePath);
      const { createReadStream } = await import('node:fs');
      const fileSize = await fileAccess.getSize(fileRef);
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(fileSize, start + Math.floor(length));
      if (end <= start) {
        return;
      }
      const chunkBytesRaw = Number(options?.chunkBytes);
      const highWaterMark = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
        ? Math.floor(chunkBytesRaw)
        : SOURCE_VERIFY_CHUNK_BYTES;
      const stream = createReadStream(fileRef, {
        start,
        end: end - 1,
        highWaterMark,
      });
      for await (const chunk of stream) {
        yield chunk;
      }
    };
    const readText = async (relativePath) => {
      const fileRef = await resolveArtifactFileReference(root, relativePath);
      const fs = await import('node:fs/promises');
      return fs.readFile(fileRef, 'utf8');
    };
    const readBinary = async (relativePath) => {
      const fileRef = await resolveArtifactFileReference(root, relativePath);
      const fs = await import('node:fs/promises');
      return fs.readFile(fileRef);
    };

    return createArtifactStorageContext({
      manifest,
      readRange,
      streamRange,
      readText,
      readBinary,
      close: fileAccess.close,
      verifyHashes: false,
    });
  } catch (error) {
    fileAccess.close().catch(() => {});
    throw error;
  }
}

async function fetchBytes(url, offset = null, length = null) {
  const headers = {};
  if (Number.isFinite(offset) && Number.isFinite(length) && length > 0) {
    const start = Math.max(0, Math.floor(offset));
    const end = start + Math.max(0, Math.floor(length)) - 1;
    headers.Range = `bytes=${start}-${end}`;
  }
  try {
    const response = await fetch(url, { headers, cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    throw createHttpArtifactFetchError(error, url, offset, length);
  }
}

function createHttpRangeBlockCache(options = {}) {
  const blockBytes = normalizeOptionalPositiveInteger(options.rangeCacheBlockBytes);
  const maxBytes = normalizeOptionalPositiveInteger(options.rangeCacheMaxBytes);
  const minBytes = normalizeOptionalPositiveInteger(options.rangeCacheMinBytes);
  if (blockBytes <= 0 || maxBytes <= 0) {
    return null;
  }

  const cache = new Map();
  const pending = new Map();
  let residentBytes = 0;

  const readCachedBlock = async (url, blockStart, blockLength) => {
    const key = `${url}\n${blockStart}`;
    const cached = cache.get(key);
    if (cached) {
      cache.delete(key);
      cache.set(key, cached);
      return cached;
    }

    let task = pending.get(key);
    if (!task) {
      task = (async () => {
        const bytes = await fetchBytes(url, blockStart, blockLength);
        if (bytes.byteLength < blockLength) {
          throw new Error(
            `HTTP range block short read for ${url}: ` +
            `offset=${blockStart}, expected=${blockLength}, got=${bytes.byteLength}`
          );
        }
        if (bytes.byteLength <= maxBytes) {
          while (residentBytes + bytes.byteLength > maxBytes && cache.size > 0) {
            const [oldestKey, oldestBytes] = cache.entries().next().value;
            cache.delete(oldestKey);
            residentBytes -= oldestBytes.byteLength;
          }
          if (residentBytes + bytes.byteLength <= maxBytes) {
            cache.set(key, bytes);
            residentBytes += bytes.byteLength;
          }
        }
        return bytes;
      })();
      pending.set(key, task);
      task.then(
        () => pending.delete(key),
        () => pending.delete(key)
      );
    }
    return task;
  };

  return async function readRange(url, fileSize, offset = 0, length = null) {
    if (
      !Number.isFinite(fileSize)
      || !Number.isFinite(offset)
      || !Number.isFinite(length)
      || length < minBytes
    ) {
      return fetchBytes(url, offset, length);
    }

    const start = Math.max(0, Math.floor(offset));
    const available = Math.max(0, Math.floor(fileSize) - start);
    const requested = Math.min(available, Math.max(0, Math.floor(length)));
    if (requested <= 0) {
      return new Uint8Array(0);
    }

    const end = start + requested;
    const chunks = [];
    let total = 0;
    let cursor = start;
    while (cursor < end) {
      const blockStart = Math.floor(cursor / blockBytes) * blockBytes;
      const blockLength = Math.min(blockBytes, Math.max(0, Math.floor(fileSize) - blockStart));
      const block = await readCachedBlock(url, blockStart, blockLength);
      const blockOffset = cursor - blockStart;
      const take = Math.min(block.byteLength - blockOffset, end - cursor);
      if (take <= 0) {
        throw new Error(
          `HTTP range block could not satisfy ${url}: offset=${cursor}, blockStart=${blockStart}`
        );
      }
      if (total === 0 && start === blockStart && take === requested && take === block.byteLength) {
        return block;
      }
      chunks.push(block.subarray(blockOffset, blockOffset + take));
      total += take;
      cursor += take;
    }

    const out = new Uint8Array(total);
    let writeOffset = 0;
    for (const chunk of chunks) {
      out.set(chunk, writeOffset);
      writeOffset += chunk.byteLength;
    }
    return out;
  };
}

export function createHttpArtifactStorageContext(baseUrl, manifest, options = {}) {
  const root = normalizeBaseUrl(baseUrl);
  if (!root || !manifest) {
    return null;
  }
  const shardSizeByPath = new Map(
    (Array.isArray(manifest.shards) ? manifest.shards : [])
      .map((shard) => [
        normalizeArtifactPath(shard?.filename),
        normalizeOptionalPositiveInteger(shard?.size, 0),
      ])
      .filter(([filename, size]) => filename && size >= 0)
  );
  const readRangeWithCache = createHttpRangeBlockCache(options);

  const readRange = async (relativePath, offset = 0, length = null) => {
    const filename = normalizeArtifactPath(relativePath);
    if (!filename) {
      throw new Error('Artifact file path is required.');
    }
    const url = `${root}/${filename}`;
    if (readRangeWithCache) {
      return readRangeWithCache(url, shardSizeByPath.get(filename) ?? null, offset, length);
    }
    return fetchBytes(url, offset, length);
  };
  const readText = async (relativePath) => {
    const filename = normalizeArtifactPath(relativePath);
    if (!filename) {
      return null;
    }
    const url = `${root}/${filename}`;
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`Failed to fetch ${filename} from ${root}: ${response.status}`);
      }
      return response.text();
    } catch (error) {
      throw createHttpArtifactFetchError(error, url);
    }
  };
  const readBinary = async (relativePath) => {
    const filename = normalizeArtifactPath(relativePath);
    if (!filename) {
      throw new Error('Artifact binary asset path is required.');
    }
    return fetchBytes(`${root}/${filename}`);
  };

  return createArtifactStorageContext({
    manifest,
    readRange,
    readText,
    readBinary,
    verifyHashes: options.verifyHashes === true
      || (options.verifyHashes !== false && getArtifactFormat(manifest) === ARTIFACT_FORMAT_DIRECT_SOURCE),
  });
}
