import {
  buildBundledTokenizer,
  buildSentencepieceTokenizer,
  inferEmbeddingOutputConfig,
  resolveConvertedAt,
  resolveManifestMoEConfig,
  resolveManifestMultimodalConfig,
} from '../converter/core.js';
import {
  normalizeQuantTag,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} from '../converter/quantization-info.js';
import { resolveEosTokenId } from '../converter/tokenizer-special-tokens.js';
import {
  getGroupType,
  parseGroupExpertIndex,
  parseGroupLayerIndex,
  resolveTensorGroup,
  resolveTensorRole,
  sortGroupIds,
} from '../formats/rdrr/index.js';
import { normalizeTensorSourceTransform } from '../formats/rdrr/source-transform-contract.js';
import { createRuntimeModelContract } from '../inference/runtime-model.js';
import { computeHash } from '../storage/shard-manager.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { encodeUtf8 } from '../formats/encode-utf8.js';
import {
  DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE,
  DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
  DIRECT_SOURCE_RUNTIME_MODE,
  DIRECT_SOURCE_RUNTIME_SCHEMA,
  DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
  normalizeAuxiliaryFiles,
  normalizeHashAlgorithm,
  normalizeHashString,
  normalizePositiveInteger,
  toPathKey,
} from './source-runtime/bundle-plan.js';

export {
  DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE,
  DIRECT_SOURCE_PATH_RUNTIME_LOCAL,
  DIRECT_SOURCE_RUNTIME_MODE,
  DIRECT_SOURCE_RUNTIME_SCHEMA,
  DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION,
  getSourceRuntimeMetadata,
} from './source-runtime/bundle-plan.js';
export { createSourceStorageContext } from './source-runtime/bundle-materialize.js';

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

function buildTransformScaleSource(transform, shardIndexByPath, tensorName) {
  const scaleSourcePath = toPathKey(transform?.scaleSourcePath);
  if (!scaleSourcePath) {
    throw new Error(
      `Source tensor "${tensorName}" sourceTransform is missing scaleSourcePath.`
    );
  }
  const shard = shardIndexByPath.get(scaleSourcePath);
  if (!Number.isInteger(shard)) {
    throw new Error(
      `Source tensor "${tensorName}" references missing scale source "${scaleSourcePath}".`
    );
  }
  return {
    shard,
    offset: normalizePositiveInteger(transform.scaleOffset, `tensor scale offset (${tensorName})`),
    size: normalizePositiveInteger(transform.scaleSize, `tensor scale size (${tensorName})`),
  };
}

function buildTransformCompanionSource(
  transform,
  shardIndexByPath,
  tensorName,
  sourcePathField,
  offsetField,
  sizeField,
  label
) {
  const sourcePath = toPathKey(transform?.[sourcePathField]);
  if (!sourcePath) {
    return null;
  }
  const shard = shardIndexByPath.get(sourcePath);
  if (!Number.isInteger(shard)) {
    throw new Error(
      `Source tensor "${tensorName}" references missing ${label} source "${sourcePath}".`
    );
  }
  return {
    shard,
    offset: normalizePositiveInteger(transform[offsetField], `tensor ${label} offset (${tensorName})`),
    size: normalizePositiveInteger(transform[sizeField], `tensor ${label} size (${tensorName})`),
  };
}

function buildSourceTensorTransform(tensor, shardIndexByPath, tensorName) {
  const transform = tensor?.sourceTransform;
  if (!transform || typeof transform !== 'object') {
    return null;
  }
  if (transform.kind === 'affine_dequant') {
    return {
      kind: transform.kind,
      scheme: transform.scheme,
      sourceDtype: transform.sourceDtype,
      targetDtype: transform.targetDtype,
      ...(transform.storageEncoding ? { storageEncoding: transform.storageEncoding } : {}),
      scale: transform.scale,
      zeroPoint: transform.zeroPoint,
    };
  }
  if (transform.kind === 'litert_rowwise_dequant') {
    const rowSumSource = buildTransformCompanionSource(
      transform,
      shardIndexByPath,
      tensorName,
      'rowSumSourcePath',
      'rowSumOffset',
      'rowSumSize',
      'row-sum'
    );
    return {
      kind: transform.kind,
      scheme: transform.scheme,
      sourceDtype: transform.sourceDtype,
      targetDtype: transform.targetDtype,
      storageEncoding: transform.storageEncoding,
      scaleSemantics: transform.scaleSemantics,
      scaleDivisor: transform.scaleDivisor,
      scaleSource: buildTransformScaleSource(transform, shardIndexByPath, tensorName),
      ...(rowSumSource
        ? {
          rowSumSource,
        }
        : {}),
    };
  }
  if (transform.kind === 'litert_axis_dequant') {
    const sumSource = buildTransformCompanionSource(
      transform,
      shardIndexByPath,
      tensorName,
      'sumSourcePath',
      'sumOffset',
      'sumSize',
      'sum'
    );
    return {
      kind: transform.kind,
      scheme: transform.scheme,
      sourceDtype: transform.sourceDtype,
      targetDtype: transform.targetDtype,
      storageEncoding: transform.storageEncoding,
      scaleSemantics: transform.scaleSemantics,
      scaleDivisor: transform.scaleDivisor,
      storageShape: transform.storageShape,
      quantAxis: transform.quantAxis,
      scaleSource: buildTransformScaleSource(transform, shardIndexByPath, tensorName),
      ...(typeof transform.scaleCompanionDtype === 'string'
        ? {
          scaleCompanionDtype: transform.scaleCompanionDtype,
        }
        : {}),
      ...(transform.scaleCompanionDequant && typeof transform.scaleCompanionDequant === 'object'
        ? {
          scaleCompanionDequant: {
            scale: transform.scaleCompanionDequant.scale,
            zeroPoint: transform.scaleCompanionDequant.zeroPoint,
          },
        }
        : {}),
      ...(sumSource
        ? {
          sumSource,
        }
        : {}),
    };
  }
  if (transform.kind === 'litert_axis_blocked_dequant') {
    const sumSource = buildTransformCompanionSource(
      transform,
      shardIndexByPath,
      tensorName,
      'sumSourcePath',
      'sumOffset',
      'sumSize',
      'sum'
    );
    return {
      kind: transform.kind,
      scheme: transform.scheme,
      sourceDtype: transform.sourceDtype,
      targetDtype: transform.targetDtype,
      storageEncoding: transform.storageEncoding,
      scaleSemantics: transform.scaleSemantics,
      scaleDivisor: transform.scaleDivisor,
      storageShape: transform.storageShape,
      quantAxis: transform.quantAxis,
      storageBlockSize: transform.storageBlockSize,
      storageLaneOrder: transform.storageLaneOrder,
      scaleSource: buildTransformScaleSource(transform, shardIndexByPath, tensorName),
      ...(sumSource
        ? {
          sumSource,
        }
        : {}),
    };
  }
  throw new Error(
    `Source tensor "${tensorName}" uses unsupported sourceTransform.kind "${transform.kind}".`
  );
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
    const sourceTransform = buildSourceTensorTransform(tensor, shardIndexByPath, name);

    const location = {
      shard,
      offset,
      size,
      shape,
      dtype,
      role,
      group,
      ...(layout ? { layout } : {}),
      ...(sourceTransform ? { sourceTransform } : {}),
    };
    if (sourceTransform) {
      normalizeTensorSourceTransform(location, name, { errorPrefix: '[SourceRuntime]' });
    }
    locations[name] = location;
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
          sourceTransform: location?.sourceTransform ?? null,
        };
      }),
    };
    group.hash = await computeHash(encodeUtf8(JSON.stringify(payload)), hashAlgorithm);
  }
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

function buildSourceRuntimeInference(options, tensorLocations) {
  const inference = cloneJsonValue(options.inference);
  if (!inference || typeof inference !== 'object' || Array.isArray(inference)) {
    throw new Error('source runtime bundle: inference config is required.');
  }
  const embeddingOutput = inferEmbeddingOutputConfig(tensorLocations);
  const embeddingPostprocessor = options.embeddingPostprocessor ?? null;
  const hasExplicitEmbeddingPostprocessor = Object.prototype.hasOwnProperty.call(
    inference?.output ?? {},
    'embeddingPostprocessor'
  );
  if (!embeddingOutput && !embeddingPostprocessor && !hasExplicitEmbeddingPostprocessor) {
    return inference;
  }
  return {
    ...inference,
    output: {
      ...inference.output,
      ...(embeddingOutput ?? {}),
      embeddingPostprocessor: hasExplicitEmbeddingPostprocessor
        ? inference.output.embeddingPostprocessor
        : embeddingPostprocessor,
    },
  };
}

function resolveSourceRuntimeTokenizer(options, rawConfig, architecture) {
  if (options.tokenizerJson) {
    return buildBundledTokenizer(
      options.tokenizerJson,
      options.tokenizerConfig ?? null,
      rawConfig
    );
  }
  if (options.tokenizerModelName || options.tokenizerModelPath) {
    const tokenizerModel = options.tokenizerModelPath || options.tokenizerModelName;
    return buildSentencepieceTokenizer(
      options.tokenizerConfig ?? null,
      rawConfig,
      architecture,
      tokenizerModel
    );
  }
  return null;
}

function resolveSourceRuntimeMetadata(options, hasTokenizer) {
  const convertedAt = resolveConvertedAt(
    options.convertedAt
    ?? options.conversionInfo?.convertedAt
    ?? null
  );
  return {
    source: 'source-runtime',
    convertedAt,
    ...(hasTokenizer ? { hasTokenizer: true } : {}),
  };
}

function buildSourceRuntimeModel(options, shards, tensorLocations, groups, quantizationInfo, manifestQuantization) {
  const rawConfig = (
    options.rawConfig
    && typeof options.rawConfig === 'object'
    && !Array.isArray(options.rawConfig)
  )
    ? options.rawConfig
    : {};
  const modelType = String(options.modelType || '').trim();
  const resolvedArchitecture = options.architecture;
  const manifestConfig = resolveManifestMultimodalConfig(rawConfig, options.manifestConfig ?? null);
  const config = {
    ...(manifestConfig.vision_config ? { vision_config: manifestConfig.vision_config } : {}),
    ...(manifestConfig.audio_config ? { audio_config: manifestConfig.audio_config } : {}),
  };
  const tokenizer = resolveSourceRuntimeTokenizer(options, rawConfig, resolvedArchitecture);
  const inference = buildSourceRuntimeInference(options, tensorLocations);
  const eosTokenId = options.eosTokenId !== undefined
    ? options.eosTokenId
    : resolveEosTokenId({
      config: rawConfig,
      generationConfig: null,
      tokenizer: options.tokenizerConfig ?? null,
      tokenizerJson: options.tokenizerJson ?? null,
    });
  const metadata = resolveSourceRuntimeMetadata(options, tokenizer != null);
  const moeConfig = modelType === 'diffusion'
    ? null
    : resolveManifestMoEConfig(
      {
        tensors: Object.entries(tensorLocations).map(([name, location]) => ({
          name,
          role: location?.role ?? null,
          layout: location?.layout ?? null,
        })),
      },
      {
        modelId: options.modelId,
        quantizationInfo,
      },
      rawConfig,
      modelType
    );

  return createRuntimeModelContract({
    sourceFormat: options.sourceKind ?? DIRECT_SOURCE_RUNTIME_MODE,
    version: 1,
    modelId: options.modelId,
    modelType,
    quantization: manifestQuantization,
    quantizationInfo,
    hashAlgorithm: normalizeHashAlgorithm(options.hashAlgorithm),
    architecture: resolvedArchitecture,
    groups,
    shards,
    totalSize: shards.reduce((sum, shard) => sum + shard.size, 0),
    tensorCount: Object.keys(tensorLocations).length,
    tensors: tensorLocations,
    tokenizer: tokenizer ?? undefined,
    moeConfig: moeConfig ?? undefined,
    ...(Object.keys(config).length > 0 ? { config } : {}),
    conversion: options.conversionInfo ?? undefined,
    inference,
    eos_token_id: eosTokenId,
    ...(rawConfig.image_token_id !== undefined ? { image_token_id: rawConfig.image_token_id } : {}),
    ...(rawConfig.audio_token_id !== undefined ? { audio_token_id: rawConfig.audio_token_id } : {}),
    ...(rawConfig.video_token_id !== undefined ? { video_token_id: rawConfig.video_token_id } : {}),
    metadata,
  });
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

  const model = buildSourceRuntimeModel(
    options,
    shards,
    tensorLocations,
    groups,
    quantizationInfo,
    manifestQuantization
  );
  if (!model.metadata || typeof model.metadata !== 'object') {
    model.metadata = {};
  }
  model.metadata.sourceRuntime = buildSourceRuntimeMetadata(
    options,
    model,
    shardSources,
    auxiliaryFiles,
    hashAlgorithm
  );

  return {
    model,
    manifest: model,
    shardSources,
  };
}
