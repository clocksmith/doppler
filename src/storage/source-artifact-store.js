import {
  createSourceStorageContext,
  getSourceRuntimeMetadata,
} from '../tooling/source-runtime-bundle.js';
import {
  computeHash,
  loadAuxText,
  loadFileFromStore,
  loadFileRangeFromStore,
  streamFileFromStore,
} from './shard-manager.js';

export function normalizeSourceArtifactPath(value) {
  return String(value || '').trim().replace(/\\/g, '/');
}

function encodeUtf8(value) {
  return new TextEncoder().encode(String(value ?? ''));
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

export function listSourceArtifactFiles(manifest) {
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
  });
}
