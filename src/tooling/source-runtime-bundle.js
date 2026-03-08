import {
  createManifest,
} from '../converter/core.js';
import {
  normalizeQuantTag,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} from '../converter/quantization-info.js';
import {
  classifyTensor,
  classifyTensorRole,
  getGroupType,
  parseGroupExpertIndex,
  parseGroupLayerIndex,
  sortGroupIds,
} from '../formats/rdrr/index.js';

const PLACEHOLDER_HASH = '0'.repeat(64);
export const DIRECT_SOURCE_RUNTIME_MODE = 'direct-source';
export const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION = 1;
export const DIRECT_SOURCE_RUNTIME_SCHEMA = `direct-source/v${DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION}`;

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

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return Math.floor(parsed);
}

function resolveTensorShape(shape, tensorName) {
  if (!Array.isArray(shape) || shape.length === 0) {
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
    fileMap.set(path, { path, size });
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
      hash: PLACEHOLDER_HASH,
      hashAlgorithm,
      offset,
    });
    shardSources.push({
      index,
      path: file.path,
      filename,
      size: file.size,
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
    const role = classifyTensorRole(name);
    const group = classifyTensor(name, modelType);
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
      hash: PLACEHOLDER_HASH,
      ...(Number.isInteger(layerIndex) ? { layerIndex } : {}),
      ...(Number.isInteger(expertIndex) ? { expertIndex } : {}),
    };
  }

  return groups;
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

  const hashAlgorithmRaw = String(options.hashAlgorithm || '').trim().toLowerCase();
  const hashAlgorithm = hashAlgorithmRaw === 'blake3' ? 'blake3' : 'sha256';
  const sourceFiles = await resolveSourceFiles(tensors, options.sourceFiles, options.resolveSourceSize);
  const { shards, shardSources } = buildSourceShards(sourceFiles, hashAlgorithm);
  const shardIndexByPath = new Map(shardSources.map((entry) => [entry.path, entry.index]));
  const tensorLocations = buildSourceTensorLocations(tensors, shardIndexByPath, modelType);
  const groups = buildSourceGroups(tensorLocations, modelType);
  const { quantizationInfo, manifestQuantization } = resolveModelQuantization(options, tensorLocations);

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
  manifest.metadata.sourceRuntime = {
    mode: DIRECT_SOURCE_RUNTIME_MODE,
    schema: DIRECT_SOURCE_RUNTIME_SCHEMA,
    schemaVersion: DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
    sourceFileCount: shardSources.length,
    invariants: {
      tensorIdentity: 'tensor.name',
      shardIdentity: 'sourcePath -> shard index',
      byteOffsets: 'shard-relative bytes',
      hashSemantics: 'placeholder shard/group hashes; verifyHashes must be false',
      cacheKeying: 'sourcePath:size',
    },
  };

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

  const shardSources = Array.isArray(options.shardSources) ? options.shardSources : null;
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
  const tokenizerJsonPath = options.tokenizerJsonPath ?? null;
  const tokenizerModelPath = options.tokenizerModelPath ?? null;
  const verifyHashes = options.verifyHashes === true;
  if (verifyHashes) {
    throw new Error(
      'source storage context: verifyHashes=true is not supported for direct-source manifests. ' +
      'Convert to persisted RDRR shards first when hash verification is required.'
    );
  }

  const loadShardRange = async (index, offset = 0, length = null) => {
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
  };

  const loadShard = async (index) => {
    const { shardSize } = resolveSourceEntry(index, manifest, shardSources);
    return loadShardRange(index, 0, shardSize);
  };

  const streamShardRange = async function* (index, offset = 0, length = null, streamOptions = {}) {
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
  };

  const loadTokenizerJson = readText && tokenizerJsonPath
    ? async () => {
      const raw = await readText(tokenizerJsonPath);
      if (typeof raw === 'string') {
        return JSON.parse(raw);
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
