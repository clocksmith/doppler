import {
  createManifest,
} from '../converter/core.js';
import {
  normalizeQuantTag,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} from '../converter/quantization-info.js';
import {
  getGroupType,
  parseGroupExpertIndex,
  parseGroupLayerIndex,
  resolveTensorGroup,
  resolveTensorRole,
  sortGroupIds,
} from '../formats/rdrr/index.js';
import { computeHash } from '../storage/shard-manager.js';

export const DIRECT_SOURCE_RUNTIME_MODE = 'direct-source';
export const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION = 1;
export const DIRECT_SOURCE_RUNTIME_SCHEMA = `direct-source/v${DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION}`;
export const DIRECT_SOURCE_PATH_RUNTIME_LOCAL = 'runtime-local';
export const DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE = 'artifact-relative';

function toPathKey(value) {
  return String(value || '').trim().replace(/\\/g, '/');
}

function toArrayBuffer(value, label) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (value instanceof Uint8Array) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function toUint8Chunk(value, label) {
  return value instanceof Uint8Array ? value : new Uint8Array(toArrayBuffer(value, label));
}

function encodeUtf8(value) {
  return new TextEncoder().encode(String(value ?? ''));
}

function normalizeHashAlgorithm(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized === 'blake3' ? 'blake3' : 'sha256';
}

function normalizeHashString(value, label) {
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

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return Math.floor(parsed);
}

function resolveTensorShape(shape, tensorName) {
  if (!Array.isArray(shape)) {
    throw new Error(`Source tensor "${tensorName}" is missing shape.`);
  }
  return shape.map((dim, index) => {
    const parsed = Number(dim);
    if (!Number.isFinite(parsed) || parsed < 1) {
      throw new Error(`Source tensor "${tensorName}" has invalid shape[${index}] (${dim}).`);
    }
    return Math.floor(parsed);
  });
}

async function resolveSourceFiles(tensors, sourceFiles, resolveSourceSize) {
  const fileMap = new Map();

  for (const entry of Array.isArray(sourceFiles) ? sourceFiles : []) {
    const path = toPathKey(entry?.path);
    if (!path) continue;
    const size = normalizePositiveInteger(entry?.size, `source file size (${path})`);
    fileMap.set(path, {
      path,
      size,
      hash: normalizeHashString(entry?.hash, `source file hash (${path})`),
      hashAlgorithm: normalizeHashAlgorithm(entry?.hashAlgorithm),
    });
  }

  for (const tensor of tensors) {
    const sourcePath = toPathKey(tensor?.sourcePath);
    if (!sourcePath) {
      throw new Error(`Source tensor "${tensor?.name ?? 'unknown'}" is missing sourcePath.`);
    }
    if (!fileMap.has(sourcePath)) {
      fileMap.set(sourcePath, { path: sourcePath, size: null });
    }
  }

  const files = Array.from(fileMap.values()).sort((left, right) => left.path.localeCompare(right.path));
  for (const file of files) {
    if (file.size != null) continue;
    if (typeof resolveSourceSize !== 'function') {
      throw new Error(
        `Source file "${file.path}" size is unknown. Provide sourceFiles[] or resolveSourceSize().`
      );
    }
    const size = await resolveSourceSize(file.path);
    file.size = normalizePositiveInteger(size, `source file size (${file.path})`);
  }

  return files;
}

function buildSourceShards(sourceFiles, hashAlgorithm) {
  const shards = [];
  const shardSources = [];
  let offset = 0;

  for (let index = 0; index < sourceFiles.length; index++) {
    const file = sourceFiles[index];
    const filename = `source_${String(index).padStart(5, '0')}.bin`;
    shards.push({
      index,
      filename,
      size: file.size,
      hash: file.hash ?? '',
      hashAlgorithm,
      offset,
    });
    shardSources.push({
      index,
      path: file.path,
      filename,
      size: file.size,
      hash: file.hash ?? '',
      hashAlgorithm,
    });
    offset += file.size;
  }

  return { shards, shardSources };
}

function buildSourceTensorLocations(tensors, shardIndexByPath, modelType) {
  const sorted = [...tensors].sort((left, right) => {
    const leftPath = toPathKey(left?.sourcePath);
    const rightPath = toPathKey(right?.sourcePath);
    const pathCmp = leftPath.localeCompare(rightPath);
    if (pathCmp !== 0) return pathCmp;
    const leftOffset = Number(left?.offset) || 0;
    const rightOffset = Number(right?.offset) || 0;
    if (leftOffset !== rightOffset) return leftOffset - rightOffset;
    return String(left?.name || '').localeCompare(String(right?.name || ''));
  });

  const locations = {};
  for (const tensor of sorted) {
    const name = String(tensor?.name || '').trim();
    if (!name) {
      throw new Error('Source tensor name is required.');
    }
    const sourcePath = toPathKey(tensor.sourcePath);
    const shard = shardIndexByPath.get(sourcePath);
    if (!Number.isInteger(shard)) {
      throw new Error(`Missing source shard mapping for tensor "${name}" (${sourcePath}).`);
    }
    const offset = normalizePositiveInteger(tensor.offset, `tensor offset (${name})`);
    const size = normalizePositiveInteger(tensor.size, `tensor size (${name})`);
    const dtype = String(tensor.dtype || '').trim().toUpperCase();
    if (!dtype) {
      throw new Error(`Source tensor "${name}" is missing dtype.`);
    }
    const shape = resolveTensorShape(tensor.shape, name);
    const role = resolveTensorRole(tensor);
    const group = resolveTensorGroup(tensor, modelType);
    const layout = typeof tensor.layout === 'string' && tensor.layout.trim()
      ? tensor.layout.trim()
      : null;

    locations[name] = {
      shard,
      offset,
      size,
      shape,
      dtype,
      role,
      group,
      ...(layout ? { layout } : {}),
    };
  }

  return locations;
}

function buildSourceGroups(tensorLocations, modelType) {
  const groupsById = new Map();

  for (const [tensorName, location] of Object.entries(tensorLocations)) {
    const groupId = String(location?.group || 'other');
    let group = groupsById.get(groupId);
    if (!group) {
      group = {
        tensors: [],
        shards: new Set(),
      };
      groupsById.set(groupId, group);
    }
    group.tensors.push(tensorName);
    if (Number.isInteger(location.shard)) {
      group.shards.add(location.shard);
    }
  }

  const groups = {};
  for (const groupId of sortGroupIds(Array.from(groupsById.keys()))) {
    const entry = groupsById.get(groupId);
    if (!entry) continue;
    const layerIndex = parseGroupLayerIndex(groupId);
    const expertIndex = parseGroupExpertIndex(groupId);
    groups[groupId] = {
      type: getGroupType(groupId, modelType),
      version: '1.0.0',
      shards: Array.from(entry.shards).sort((left, right) => left - right),
      tensors: [...entry.tensors].sort((left, right) => left.localeCompare(right)),
      hash: '',
      ...(Number.isInteger(layerIndex) ? { layerIndex } : {}),
      ...(Number.isInteger(expertIndex) ? { expertIndex } : {}),
    };
  }

  return groups;
}

async function assignGroupHashes(groups, tensorLocations, hashAlgorithm) {
  const groupIds = sortGroupIds(Object.keys(groups ?? {}));
  for (const groupId of groupIds) {
    const group = groups[groupId];
    if (!group) continue;
    const tensors = Array.isArray(group.tensors) ? group.tensors : [];
    const payload = {
      groupId,
      type: group.type ?? null,
      version: group.version ?? null,
      layerIndex: Number.isInteger(group.layerIndex) ? group.layerIndex : null,
      expertIndex: Number.isInteger(group.expertIndex) ? group.expertIndex : null,
      tensors: tensors.map((tensorName) => {
        const location = tensorLocations?.[tensorName] ?? null;
        return {
          name: tensorName,
          shard: location?.shard ?? null,
          offset: location?.offset ?? null,
          size: location?.size ?? null,
          dtype: location?.dtype ?? null,
          shape: Array.isArray(location?.shape) ? location.shape : null,
          layout: location?.layout ?? null,
        };
      }),
    };
    group.hash = await computeHash(encodeUtf8(JSON.stringify(payload)), hashAlgorithm);
  }
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

function normalizeAuxiliaryFiles(auxiliaryFiles, defaultHashAlgorithm) {
  const normalized = [];
  for (const entry of Array.isArray(auxiliaryFiles) ? auxiliaryFiles : []) {
    const resolved = normalizeAuxiliaryFileEntry(entry, defaultHashAlgorithm);
    if (resolved) normalized.push(resolved);
  }
  normalized.sort((left, right) => left.path.localeCompare(right.path));
  return normalized;
}

function buildSourceRuntimeMetadata(options, manifest, shardSources, auxiliaryFiles, hashAlgorithm) {
  const tokenizerJsonPath = typeof options.tokenizerJsonPath === 'string' && options.tokenizerJsonPath.trim()
    ? toPathKey(options.tokenizerJsonPath)
    : null;
  const tokenizerConfigPath = typeof options.tokenizerConfigPath === 'string' && options.tokenizerConfigPath.trim()
    ? toPathKey(options.tokenizerConfigPath)
    : null;
  const tokenizerModelPath = typeof options.tokenizerModelPath === 'string' && options.tokenizerModelPath.trim()
    ? toPathKey(options.tokenizerModelPath)
    : null;
  const hasFullSourceDigests = shardSources.every((entry) => typeof entry.hash === 'string' && entry.hash.length > 0);
  const hasFullAuxDigests = auxiliaryFiles.every((entry) => typeof entry.hash === 'string' && entry.hash.length > 0);

  return {
    mode: DIRECT_SOURCE_RUNTIME_MODE,
    schema: DIRECT_SOURCE_RUNTIME_SCHEMA,
    schemaVersion: DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
    sourceKind: typeof options.sourceKind === 'string' && options.sourceKind.trim()
      ? String(options.sourceKind).trim().toLowerCase()
      : null,
    hashAlgorithm,
    pathSemantics: DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
    sourceFileCount: shardSources.length,
    auxiliaryFileCount: auxiliaryFiles.length,
    sourceFiles: shardSources.map((entry) => ({
      index: entry.index,
      path: entry.path,
      filename: entry.filename,
      size: entry.size,
      hash: entry.hash,
      hashAlgorithm: entry.hashAlgorithm,
    })),
    auxiliaryFiles,
    tokenizer: {
      jsonPath: tokenizerJsonPath,
      configPath: tokenizerConfigPath,
      modelPath: tokenizerModelPath,
    },
    invariants: {
      tensorIdentity: 'tensor.name',
      shardIdentity: 'sourceFiles[index].path',
      byteOffsets: 'shard-relative bytes',
      hashSemantics: hasFullSourceDigests && hasFullAuxDigests
        ? 'sourceFiles[*].hash digests raw source files; auxiliaryFiles[*].hash digests config/index/tokenizer assets'
        : 'source digests are incomplete; persist a materialized direct-source manifest before release claims',
      cacheKeying: hasFullSourceDigests ? 'path:size:hash' : 'path:size',
      tokenizerAssetsCovered: tokenizerJsonPath != null || tokenizerModelPath != null,
      manifestFamily: manifest?.modelType ?? null,
    },
  };
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

function resolveModelQuantization(options, tensorLocations) {
  const sourceQuantization = options.sourceQuantization
    ? normalizeQuantTag(options.sourceQuantization)
    : null;
  const tensorEntries = Object.entries(tensorLocations).map(([name, location]) => ({
    name,
    dtype: location?.dtype ?? null,
    role: location?.role ?? null,
    layout: location?.layout ?? null,
  }));
  const effectiveQuantizationInfo = resolveEffectiveQuantizationInfo(
    options.quantizationInfo ?? null,
    tensorEntries
  );
  const fallbackManifestQuantization = sourceQuantization
    ? resolveManifestQuantization(sourceQuantization, sourceQuantization.toUpperCase())
    : 'F16';
  const manifestQuantization = options.manifestQuantization
    ?? resolveManifestQuantization(
      effectiveQuantizationInfo.weights,
      fallbackManifestQuantization
    );
  return {
    quantizationInfo: effectiveQuantizationInfo,
    manifestQuantization,
  };
}

export async function buildSourceRuntimeBundle(options = {}) {
  const modelId = String(options.modelId || '').trim();
  if (!modelId) {
    throw new Error('source runtime bundle: modelId is required.');
  }

  const modelType = String(options.modelType || '').trim();
  if (!modelType) {
    throw new Error('source runtime bundle: modelType is required.');
  }

  const inference = options.inference;
  if (!inference || typeof inference !== 'object') {
    throw new Error('source runtime bundle: inference config is required.');
  }

  if (modelType !== 'diffusion') {
    const architecture = options.architecture;
    if (!architecture || typeof architecture !== 'object') {
      throw new Error(
        'source runtime bundle: architecture object is required for non-diffusion modelType.'
      );
    }
  }

  const tensors = Array.isArray(options.tensors) ? options.tensors : null;
  if (!tensors || tensors.length === 0) {
    throw new Error('source runtime bundle: tensors[] is required.');
  }

  const hashAlgorithm = normalizeHashAlgorithm(options.hashAlgorithm);
  const sourceFiles = await resolveSourceFiles(tensors, options.sourceFiles, options.resolveSourceSize);
  const { shards, shardSources } = buildSourceShards(sourceFiles, hashAlgorithm);
  const shardIndexByPath = new Map(shardSources.map((entry) => [entry.path, entry.index]));
  const tensorLocations = buildSourceTensorLocations(tensors, shardIndexByPath, modelType);
  const groups = buildSourceGroups(tensorLocations, modelType);
  await assignGroupHashes(groups, tensorLocations, hashAlgorithm);
  const { quantizationInfo, manifestQuantization } = resolveModelQuantization(options, tensorLocations);
  const auxiliaryFiles = normalizeAuxiliaryFiles(options.auxiliaryFiles, hashAlgorithm);

  const model = {
    name: options.modelName || modelId,
    modelId,
    modelType,
    tensors: tensors.map((tensor) => ({
      name: tensor.name,
      shape: tensor.shape,
      dtype: tensor.dtype,
      size: tensor.size,
      offset: tensor.offset,
      sourcePath: tensor.sourcePath,
    })),
    config: options.rawConfig ?? {},
    architecture: options.architectureHint ?? modelType,
    quantization: manifestQuantization,
    tokenizerJson: options.tokenizerJson ?? null,
    tokenizerConfig: options.tokenizerConfig ?? null,
    tokenizerModel: options.tokenizerModelName ?? null,
    embeddingPostprocessor: options.embeddingPostprocessor ?? null,
  };

  const manifest = createManifest(modelId, model, shards, tensorLocations, {
    source: 'source-runtime',
    modelType,
    quantization: manifestQuantization,
    quantizationInfo,
    hashAlgorithm,
    architecture: options.architecture,
    inference,
    eosTokenId: options.eosTokenId,
    convertedAt: options.convertedAt ?? null,
    conversionInfo: options.conversionInfo ?? null,
  });

  manifest.groups = groups;
  if (!manifest.metadata || typeof manifest.metadata !== 'object') {
    manifest.metadata = {};
  }
  manifest.metadata.sourceRuntime = buildSourceRuntimeMetadata(
    options,
    manifest,
    shardSources,
    auxiliaryFiles,
    hashAlgorithm
  );

  return {
    manifest,
    shardSources,
  };
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
  const manifest = options.manifest;
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('source storage context: manifest is required.');
  }

  const sourceRuntime = getSourceRuntimeMetadata(manifest);
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
  const auxiliaryFileMap = new Map(
    (sourceRuntime?.auxiliaryFiles ?? []).map((entry) => [entry.path, entry])
  );
  const tokenizerJsonPath = options.tokenizerJsonPath ?? sourceRuntime?.tokenizer?.jsonPath ?? null;
  const tokenizerModelPath = options.tokenizerModelPath ?? sourceRuntime?.tokenizer?.modelPath ?? null;
  const verifyHashes = options.verifyHashes === true;
  const allowRangeFastPath = verifyHashes !== true;

  const loadShardRange = allowRangeFastPath ? async (index, offset = 0, length = null) => {
    const { sourcePath, shardSize } = resolveSourceEntry(index, manifest, shardSources);
    const start = normalizePositiveInteger(offset, `shard offset (${index})`);
    const maxLength = Math.max(0, shardSize - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `shard length (${index})`));
    if (requested <= 0) {
      return new ArrayBuffer(0);
    }
    const payload = await readRange(sourcePath, start, requested);
    return toArrayBuffer(payload, `readRange(${sourcePath})`);
  } : null;

  const loadShard = async (index) => {
    const { shardSize } = resolveSourceEntry(index, manifest, shardSources);
    if (loadShardRange) {
      return loadShardRange(index, 0, shardSize);
    }
    const { sourcePath } = resolveSourceEntry(index, manifest, shardSources);
    const payload = await readRange(sourcePath, 0, shardSize);
    return toArrayBuffer(payload, `readRange(${sourcePath})`);
  };

  const streamShardRange = allowRangeFastPath ? async function* (index, offset = 0, length = null, streamOptions = {}) {
    const { sourcePath, shardSize } = resolveSourceEntry(index, manifest, shardSources);
    const start = normalizePositiveInteger(offset, `shard stream offset (${index})`);
    const maxLength = Math.max(0, shardSize - start);
    const requested = length == null
      ? maxLength
      : Math.min(maxLength, normalizePositiveInteger(length, `shard stream length (${index})`));
    if (requested <= 0) {
      return;
    }

    if (streamRange) {
      for await (const chunk of streamRange(sourcePath, start, requested, streamOptions)) {
        yield toUint8Chunk(chunk, `streamRange(${sourcePath})`);
      }
      return;
    }

    const chunkBytesRaw = Number(streamOptions?.chunkBytes);
    const chunkBytes = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
      ? Math.floor(chunkBytesRaw)
      : 4 * 1024 * 1024;
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
  } : null;

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

  return {
    loadShard,
    loadShardRange,
    streamShardRange,
    loadTokenizerJson,
    loadTokenizerModel,
    verifyHashes,
  };
}
