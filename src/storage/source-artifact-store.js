import {
  createSourceStorageContext,
  DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
  DIRECT_SOURCE_RUNTIME_MODE,
  DIRECT_SOURCE_RUNTIME_SCHEMA,
  DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
  getSourceRuntimeMetadata,
} from '../tooling/source-runtime-bundle.js';
import {
  computeHash,
  loadAuxText,
  loadFileFromStore,
  loadFileRangeFromStore,
  streamFileFromStore,
} from './shard-manager.js';
import { cloneJsonValue } from '../utils/clone-json.js';
import { encodeUtf8 } from '../utils/encode-utf8.js';

export function normalizeSourceArtifactPath(value) {
  return String(value || '').trim().replace(/\\/g, '/');
}

function normalizeStoredHash(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

function normalizeStoredHashAlgorithm(value, fallbackAlgorithm = null) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (normalized) {
    return normalized;
  }
  if (typeof fallbackAlgorithm === 'string' && fallbackAlgorithm.trim()) {
    return fallbackAlgorithm.trim().toLowerCase();
  }
  return null;
}

function resolveStoredTokenizerPaths(manifest) {
  const tokenizer = manifest?.tokenizer;
  if (!tokenizer || typeof tokenizer !== 'object') {
    return {
      jsonPath: null,
      configPath: null,
      modelPath: null,
    };
  }
  const tokenizerType = typeof tokenizer.type === 'string' ? tokenizer.type.trim().toLowerCase() : '';
  const hasBundledTokenizerJson = (
    (tokenizerType === 'bundled' || tokenizerType === 'huggingface')
    && typeof tokenizer.file === 'string'
    && tokenizer.file.trim().length > 0
  );
  const sentencepieceModel = typeof tokenizer.sentencepieceModel === 'string'
    ? tokenizer.sentencepieceModel.trim()
    : '';
  const hasTokenizerModel = tokenizerType === 'sentencepiece' || sentencepieceModel.length > 0;
  return {
    // ShardManager persists tokenizer assets under canonical store-local filenames.
    jsonPath: hasBundledTokenizerJson ? 'tokenizer.json' : null,
    configPath: null,
    modelPath: hasTokenizerModel ? 'tokenizer.model' : null,
  };
}

export function synthesizeStoredSourceArtifactManifest(manifest) {
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    return {
      manifest,
      changed: false,
    };
  }
  if (getSourceRuntimeMetadata(manifest)) {
    return {
      manifest,
      changed: false,
    };
  }

  const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
  const manifestHashAlgorithm = normalizeStoredHashAlgorithm(manifest.hashAlgorithm);
  if (!manifestHashAlgorithm || shards.length === 0) {
    return {
      manifest,
      changed: false,
    };
  }

  const sourceFiles = [];
  for (let index = 0; index < shards.length; index += 1) {
    const shard = shards[index];
    const filename = normalizeSourceArtifactPath(shard?.filename);
    if (!filename || !Number.isFinite(shard?.size)) {
      return {
        manifest,
        changed: false,
      };
    }
    sourceFiles.push({
      index,
      path: filename,
      filename,
      size: Math.max(0, Math.floor(Number(shard.size))),
      hash: normalizeStoredHash(shard?.hash ?? shard?.blake3 ?? shard?.sha256 ?? shard?.digest),
      hashAlgorithm: normalizeStoredHashAlgorithm(shard?.hashAlgorithm, manifestHashAlgorithm),
    });
  }

  const tokenizerPaths = resolveStoredTokenizerPaths(manifest);
  const nextManifest = cloneJsonValue(manifest);
  if (!nextManifest.metadata || typeof nextManifest.metadata !== 'object' || Array.isArray(nextManifest.metadata)) {
    nextManifest.metadata = {};
  }
  nextManifest.metadata.sourceRuntime = {
    mode: DIRECT_SOURCE_RUNTIME_MODE,
    schema: DIRECT_SOURCE_RUNTIME_SCHEMA,
    schemaVersion: DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
    sourceKind: 'rdrr',
    hashAlgorithm: manifestHashAlgorithm,
    pathSemantics: DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
    sourceFileCount: sourceFiles.length,
    auxiliaryFileCount: 0,
    sourceFiles,
    auxiliaryFiles: [],
    tokenizer: tokenizerPaths,
    invariants: {
      tensorIdentity: 'manifest.tensors',
      shardIdentity: 'manifest.shards[index].filename',
      byteOffsets: 'shard-relative bytes',
      hashSemantics: 'sourceFiles[*].hash mirrors stored shard digests validated at import time',
      cacheKeying: 'path:size:hash',
      tokenizerAssetsCovered: Boolean(tokenizerPaths.jsonPath || tokenizerPaths.modelPath),
      manifestFamily: manifest?.modelType ?? null,
    },
  };

  return {
    manifest: nextManifest,
    changed: true,
  };
}

function normalizeArtifactFile(entry, kind) {
  const path = normalizeSourceArtifactPath(entry?.path);
  if (!path) {
    return null;
  }
  return {
    path,
    size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : null,
    hash: typeof entry?.hash === 'string' && entry.hash.trim() ? entry.hash.trim().toLowerCase() : null,
    hashAlgorithm: typeof entry?.hashAlgorithm === 'string' && entry.hashAlgorithm.trim()
      ? entry.hashAlgorithm.trim().toLowerCase()
      : null,
    kind,
  };
}

function pushArtifactFile(files, seen, entry, kind) {
  const normalized = normalizeArtifactFile(entry, kind);
  if (!normalized || seen.has(normalized.path)) {
    return;
  }
  seen.add(normalized.path);
  files.push(normalized);
}

function collectSourceArtifactFiles(sourceRuntime) {
  if (!sourceRuntime) {
    return {
      sourceFiles: [],
      auxiliaryFiles: [],
      files: [],
    };
  }

  const files = [];
  const seen = new Set();
  const sourceFiles = [];

  for (let index = 0; index < sourceRuntime.sourceFiles.length; index += 1) {
    const entry = sourceRuntime.sourceFiles[index];
    const normalized = normalizeArtifactFile(entry, 'source');
    if (!normalized) {
      continue;
    }
    sourceFiles.push({
      ...normalized,
      index: Number.isFinite(entry?.index) ? Math.max(0, Math.floor(Number(entry.index))) : index,
    });
    pushArtifactFile(files, seen, entry, 'source');
  }
  for (const entry of sourceRuntime.auxiliaryFiles) {
    pushArtifactFile(files, seen, entry, entry?.kind || 'auxiliary');
  }

  const tokenizer = sourceRuntime.tokenizer ?? {};
  const auxiliaryByPath = new Map(
    sourceRuntime.auxiliaryFiles.map((entry) => [normalizeSourceArtifactPath(entry?.path), entry])
  );
  for (const [path, kind] of [
    [tokenizer.jsonPath, 'tokenizer_json'],
    [tokenizer.configPath, 'tokenizer_config'],
    [tokenizer.modelPath, 'tokenizer_model'],
  ]) {
    const normalizedPath = normalizeSourceArtifactPath(path);
    if (!normalizedPath) {
      continue;
    }
    pushArtifactFile(files, seen, auxiliaryByPath.get(normalizedPath) ?? { path: normalizedPath }, kind);
  }

  files.sort((left, right) => left.path.localeCompare(right.path));
  return {
    sourceFiles,
    auxiliaryFiles: files.filter((entry) => entry.kind !== 'source'),
    files,
  };
}

function listSourceArtifactFiles(manifest) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  return collectSourceArtifactFiles(sourceRuntime).files;
}

export function resolveSourceArtifact(manifest) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (!sourceRuntime) {
    return null;
  }
  const { sourceFiles, auxiliaryFiles, files } = collectSourceArtifactFiles(sourceRuntime);
  const totalBytes = files.reduce((sum, entry) => sum + (entry.size || 0), 0);
  return {
    sourceRuntime,
    sourceFiles,
    auxiliaryFiles,
    files,
    totalBytes,
    fingerprint: JSON.stringify({
      mode: sourceRuntime.mode,
      schema: sourceRuntime.schema,
      hashAlgorithm: sourceRuntime.hashAlgorithm,
      pathSemantics: sourceRuntime.pathSemantics,
      sourceKind: sourceRuntime.sourceKind,
      files: files.map((entry) => ({
        path: entry.path,
        size: entry.size,
        hash: entry.hash,
        hashAlgorithm: entry.hashAlgorithm,
        kind: entry.kind,
      })),
    }),
  };
}

export function buildSourceArtifactFingerprint(manifest) {
  return resolveSourceArtifact(manifest)?.fingerprint ?? null;
}

async function loadStoreFile(path) {
  try {
    return await loadFileFromStore(path);
  } catch (error) {
    const message = String(error?.message || '');
    if (error?.name === 'NotFoundError' || message.toLowerCase().includes('not found')) {
      return null;
    }
    throw error;
  }
}

export async function verifyStoredSourceArtifact(manifest, options = {}) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (!sourceRuntime) {
    throw new Error('verifyStoredSourceArtifact requires a direct-source manifest.');
  }

  const checkHashes = options.checkHashes !== false;
  const missingFiles = [];
  const corruptFiles = [];
  const files = listSourceArtifactFiles(manifest);

  for (const entry of files) {
    const payload = await loadStoreFile(entry.path);
    if (!(payload instanceof ArrayBuffer)) {
      missingFiles.push(entry.path);
      continue;
    }
    if (!checkHashes || !entry.hash) {
      continue;
    }
    const isTextAsset = entry.kind === 'config'
      || entry.kind === 'tokenizer_json'
      || entry.kind === 'tokenizer_config'
      || entry.kind === 'safetensors_index';
    const computedHash = isTextAsset
      ? await computeHash(encodeUtf8(new TextDecoder().decode(payload)), entry.hashAlgorithm || sourceRuntime.hashAlgorithm)
      : await computeHash(new Uint8Array(payload), entry.hashAlgorithm || sourceRuntime.hashAlgorithm);
    if (computedHash !== entry.hash) {
      corruptFiles.push(entry.path);
    }
  }

  return {
    valid: missingFiles.length === 0 && (!checkHashes || corruptFiles.length === 0),
    missingFiles,
    corruptFiles,
  };
}

export function createStoredSourceArtifactContext(manifest, options = {}) {
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (!sourceRuntime) {
    throw new Error('createStoredSourceArtifactContext requires a direct-source manifest.');
  }

  const readRange = async (path, offset, length) => loadFileRangeFromStore(path, offset, length);
  const streamRange = (path, offset, length, streamOptions = {}) => {
    const stream = streamFileFromStore(path, {
      chunkBytes: streamOptions?.chunkBytes,
      offset,
      length,
    });
    if (!stream) {
      return null;
    }
    return stream;
  };
  const readText = async (path) => loadAuxText(path);
  const readBinary = async (path) => {
    const payload = await loadStoreFile(path);
    if (!(payload instanceof ArrayBuffer)) {
      throw new Error(`Missing stored source binary file: ${path}`);
    }
    return payload;
  };

  return createSourceStorageContext({
    manifest,
    readRange,
    streamRange: streamRange ? (async function* (path, offset, length, streamOptions = {}) {
      const stream = streamRange(path, offset, length, streamOptions);
      if (!stream) {
        const payload = await loadFileRangeFromStore(path, offset, length);
        yield new Uint8Array(payload);
        return;
      }
      for await (const chunk of stream) {
        yield chunk;
      }
    }) : null,
    readText,
    readBinary,
    verifyHashes: options.verifyHashes !== false,
    // Stored shard/source assets were already hash-verified at import time. Warm loads
    // should not re-stream the full files just to re-prove the same digest.
    sourceHashesTrusted: true,
  });
}
