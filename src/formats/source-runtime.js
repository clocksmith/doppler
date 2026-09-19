export const DIRECT_SOURCE_RUNTIME_MODE = 'direct-source';
export const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION = 1;
export const DIRECT_SOURCE_RUNTIME_SCHEMA = `direct-source/v${DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION}`;
export const DIRECT_SOURCE_PATH_RUNTIME_LOCAL = 'runtime-local';
export const DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE = 'artifact-relative';

export function toPathKey(value) {
  return String(value || '').trim().replace(/\\/g, '/');
}

export function normalizeHashAlgorithm(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized === 'blake3' ? 'blake3' : 'sha256';
}

export function normalizeHashString(value, label) {
  if (value == null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) return null;
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`${label} must be a 64-character lowercase hex digest.`);
  }
  return normalized;
}

function normalizeAssetKind(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (!normalized) return 'unknown';
  return normalized;
}

export function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return Math.floor(parsed);
}

function normalizeAuxiliaryFileEntry(entry, defaultHashAlgorithm) {
  const path = toPathKey(entry?.path);
  if (!path) return null;
  return {
    path,
    size: normalizePositiveInteger(entry?.size, `source auxiliary file size (${path})`),
    hash: normalizeHashString(entry?.hash, `source auxiliary file hash (${path})`),
    hashAlgorithm: normalizeHashAlgorithm(entry?.hashAlgorithm ?? defaultHashAlgorithm),
    kind: normalizeAssetKind(entry?.kind),
  };
}

export function normalizeAuxiliaryFiles(auxiliaryFiles, defaultHashAlgorithm) {
  const normalized = [];
  for (const entry of Array.isArray(auxiliaryFiles) ? auxiliaryFiles : []) {
    const resolved = normalizeAuxiliaryFileEntry(entry, defaultHashAlgorithm);
    if (resolved) normalized.push(resolved);
  }
  normalized.sort((left, right) => left.path.localeCompare(right.path));
  return normalized;
}

export function getSourceRuntimeMetadata(manifest) {
  const metadata = manifest?.metadata?.sourceRuntime;
  if (!metadata || typeof metadata !== 'object') {
    return null;
  }
  if (metadata.mode !== DIRECT_SOURCE_RUNTIME_MODE) {
    return null;
  }

  const hashAlgorithm = normalizeHashAlgorithm(metadata.hashAlgorithm);
  const sourceFiles = Array.isArray(metadata.sourceFiles)
    ? metadata.sourceFiles
      .map((entry) => {
        const path = toPathKey(entry?.path);
        if (!path) return null;
        return {
          index: normalizePositiveInteger(entry?.index ?? 0, `source runtime sourceFiles index (${path})`),
          path,
          filename: typeof entry?.filename === 'string' && entry.filename.trim()
            ? entry.filename.trim()
            : null,
          size: normalizePositiveInteger(entry?.size, `source runtime sourceFiles size (${path})`),
          hash: normalizeHashString(entry?.hash, `source runtime sourceFiles hash (${path})`),
          hashAlgorithm: normalizeHashAlgorithm(entry?.hashAlgorithm ?? hashAlgorithm),
        };
      })
      .filter(Boolean)
      .sort((left, right) => left.index - right.index)
    : [];
  const auxiliaryFiles = normalizeAuxiliaryFiles(metadata.auxiliaryFiles, hashAlgorithm);
  const tokenizer = metadata.tokenizer && typeof metadata.tokenizer === 'object'
    ? {
      jsonPath: typeof metadata.tokenizer.jsonPath === 'string' && metadata.tokenizer.jsonPath.trim()
        ? toPathKey(metadata.tokenizer.jsonPath)
        : null,
      configPath: typeof metadata.tokenizer.configPath === 'string' && metadata.tokenizer.configPath.trim()
        ? toPathKey(metadata.tokenizer.configPath)
        : null,
      modelPath: typeof metadata.tokenizer.modelPath === 'string' && metadata.tokenizer.modelPath.trim()
        ? toPathKey(metadata.tokenizer.modelPath)
        : null,
    }
    : { jsonPath: null, configPath: null, modelPath: null };

  return {
    mode: DIRECT_SOURCE_RUNTIME_MODE,
    schema: DIRECT_SOURCE_RUNTIME_SCHEMA,
    schemaVersion: DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
    sourceKind: typeof metadata.sourceKind === 'string' && metadata.sourceKind.trim()
      ? String(metadata.sourceKind).trim().toLowerCase()
      : null,
    hashAlgorithm,
    pathSemantics: metadata.pathSemantics === DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE
      ? DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE
      : DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
    sourceFiles,
    auxiliaryFiles,
    tokenizer,
  };
}
