import {
  createWeightBuffer,
  createCpuWeightBuffer,
  getWeightDtype,
  isWeightBuffer,
  isCpuWeightBuffer,
  isGpuBufferInstance,
} from '../gpu/weight-buffer.js';
import { log } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { loadTensorRange } from './tensors/tensor-reader.js';

const EMBED_TENSOR_CANDIDATES = [
  'model.language_model.embed_tokens_per_layer.weight',
  'language_model.embed_tokens_per_layer.weight',
  'language_model.model.embed_tokens_per_layer.weight',
  'model.embed_tokens_per_layer.weight',
  'embed_tokens_per_layer.weight',
];

const SPLIT_EMBED_TENSOR_CANDIDATE_FACTORIES = [
  (layerIndex) => `model.language_model.layers.${layerIndex}.embed_tokens_per_layer.weight`,
  (layerIndex) => `language_model.layers.${layerIndex}.embed_tokens_per_layer.weight`,
  (layerIndex) => `language_model.model.layers.${layerIndex}.embed_tokens_per_layer.weight`,
  (layerIndex) => `model.layers.${layerIndex}.embed_tokens_per_layer.weight`,
  (layerIndex) => `layers.${layerIndex}.embed_tokens_per_layer.weight`,
];

const PROJECTION_TENSOR_CANDIDATES = [
  'model.language_model.per_layer_model_projection.weight',
  'language_model.per_layer_model_projection.weight',
  'language_model.model.per_layer_model_projection.weight',
  'model.per_layer_model_projection.weight',
  'per_layer_model_projection.weight',
];

const PROJECTION_NORM_TENSOR_CANDIDATES = [
  'model.language_model.per_layer_projection_norm.weight',
  'language_model.per_layer_projection_norm.weight',
  'language_model.model.per_layer_projection_norm.weight',
  'model.per_layer_projection_norm.weight',
  'per_layer_projection_norm.weight',
];

function wrapRawTensorAsWeightBuffer(ctx, tensor, name) {
  if (tensor == null || isWeightBuffer(tensor)) {
    return tensor;
  }
  const location = ctx.tensorLocations.get(name) ?? null;
  if (!location?.shape || location.shape.length !== 2) {
    return tensor;
  }
  const layout = ctx.resolveWeightLayout(location);
  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  if (tensor instanceof Float32Array) {
    return createCpuWeightBuffer(tensor, dtype, layout, location.shape, name);
  }
  if (!isGpuBufferInstance(tensor)) {
    return tensor;
  }
  return createWeightBuffer(tensor, dtype, layout, location.shape, name);
}

function createRangeBackedTensorSource(ctx, name, location) {
  if (typeof ctx.loadShardRange !== 'function') {
    return null;
  }
  const normalizedLocationDtype = typeof location?.dtype === 'string'
    ? location.dtype.toLowerCase()
    : 'f32';
  return {
    kind: 'tensor_range_source',
    sourceDtype: normalizedLocationDtype,
    async loadRange(byteOffset, byteLength) {
      return loadTensorRange(location, name, byteOffset, byteLength, ctx.loadShardRange);
    },
  };
}

function createRangeBackedWeightBuffer(ctx, name, location) {
  const source = createRangeBackedTensorSource(ctx, name, location);
  if (!source || !location?.shape || location.shape.length !== 2) {
    return null;
  }
  const layout = ctx.resolveWeightLayout(location);
  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  return createCpuWeightBuffer(source, dtype, layout, location.shape, name);
}

function getExpectedTensorLogicalByteLength(location) {
  const shape = Array.isArray(location?.shape) ? location.shape : null;
  if (!shape || shape.length === 0) {
    return null;
  }
  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', {
    dtype,
  });
  const elementCount = shape.reduce((total, dimension) => {
    const parsed = Number(dimension);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return NaN;
    }
    return total * parsed;
  }, 1);
  if (!Number.isFinite(elementCount) || elementCount <= 0) {
    return null;
  }
  return elementCount * bytesPerElement;
}

function getLoadedTensorResidentByteLength(tensor) {
  if (isWeightBuffer(tensor)) {
    const bufferSize = Number(tensor.buffer?.size);
    return Number.isFinite(bufferSize) && bufferSize > 0 ? bufferSize : null;
  }
  if (isGpuBufferInstance(tensor)) {
    const bufferSize = Number(tensor.size);
    return Number.isFinite(bufferSize) && bufferSize > 0 ? bufferSize : null;
  }
  if (ArrayBuffer.isView(tensor)) {
    return tensor.byteLength;
  }
  if (tensor instanceof ArrayBuffer) {
    return tensor.byteLength;
  }
  if (typeof tensor?.byteLength === 'number' && Number.isFinite(tensor.byteLength) && tensor.byteLength > 0) {
    return tensor.byteLength;
  }
  return null;
}

function isPackedResidentWeightTensor(tensor) {
  if (!isWeightBuffer(tensor)) {
    return false;
  }
  const dtype = String(getWeightDtype(tensor) || '').trim().toLowerCase();
  if (!dtype) {
    return false;
  }
  return dtype !== 'f16' && dtype !== 'f32' && dtype !== 'bf16';
}

function isPackedQuantizedLocation(location) {
  const dtype = String(location?.dtype || '').trim().toLowerCase();
  if (!dtype) {
    return false;
  }
  return dtype !== 'f16' && dtype !== 'f32' && dtype !== 'bf16';
}

function validateResidentPerLayerProjectionTensor(ctx, name, location, tensor) {
  if (
    isPackedResidentWeightTensor(tensor)
    || (isPackedQuantizedLocation(location) && (isGpuBufferInstance(tensor) || isWeightBuffer(tensor)))
  ) {
    return tensor;
  }
  const expectedBytes = getExpectedTensorLogicalByteLength(location);
  const residentBytes = getLoadedTensorResidentByteLength(tensor);
  if (
    !Number.isFinite(expectedBytes)
    || !Number.isFinite(residentBytes)
    || residentBytes >= expectedBytes
  ) {
    return tensor;
  }

  const rangeBacked = createRangeBackedWeightBuffer(ctx, name, location);
  if (rangeBacked) {
    log.warn(
      'Loader',
      `Per-layer input projection "${name}" materialized to ${residentBytes} bytes, ` +
      `but its declared shape/dtype requires ${expectedBytes} bytes. ` +
      'Falling back to range-backed CPU source to preserve manifest tensor contract.'
    );
    return rangeBacked;
  }

  throw new Error(
    `Manifest "${ctx.modelId ?? 'unknown'}" resolved per-layer input projection "${name}" ` +
    `to ${residentBytes} resident bytes, but its declared shape/dtype requires ${expectedBytes}. ` +
    'Range-backed shard loading is unavailable, so this direct-source tensor cannot be materialized safely.'
  );
}

function resolvePerLayerInputMaterializationMode(ctx, label, name, location) {
  if (label !== 'embedTokensPerLayer') {
    return null;
  }
  const sessionConfig = ctx.perLayerInputSession;
  if (!sessionConfig || typeof sessionConfig !== 'object') {
    throw new Error(
      `Manifest "${ctx.modelId ?? 'unknown'}" requires per-layer input session policy ` +
      'before loading embedTokensPerLayer.'
    );
  }

  const mode = sessionConfig.materialization;
  if (mode === 'auto') {
    const shouldStream = location && typeof ctx.shouldStreamLargeWeight === 'function'
      ? ctx.shouldStreamLargeWeight(name, location, label)
      : false;
    return shouldStream ? 'range_backed' : 'gpu_resident';
  }
  if (
    mode === 'range_backed'
    || mode === 'cpu_resident'
    || mode === 'gpu_resident'
    || mode === 'gpu_split_tables'
  ) {
    return mode;
  }
  throw new Error(
    `Manifest "${ctx.modelId ?? 'unknown'}" has unsupported per-layer input materialization ` +
    `"${String(mode)}".`
  );
}

function resolveSplitPerLayerEmbedTensorNames(tensorLocations, numLayers) {
  if (!(tensorLocations instanceof Map)) {
    return null;
  }
  if (!Number.isInteger(numLayers) || numLayers <= 0) {
    return null;
  }
  const splitNamePattern = /\.layers\.\d+\.embed_tokens_per_layer\.weight$/;
  const hasAnySplitTensor = Array.from(tensorLocations.keys()).some((name) => splitNamePattern.test(name));
  const names = [];
  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    let resolvedName = null;
    for (const createName of SPLIT_EMBED_TENSOR_CANDIDATE_FACTORIES) {
      const candidate = createName(layerIndex);
      if (tensorLocations.has(candidate)) {
        resolvedName = candidate;
        break;
      }
    }
    if (!resolvedName) {
      if (hasAnySplitTensor) {
        throw new Error(
          `Manifest split per-layer input table set is incomplete. ` +
          `Missing layer ${layerIndex} embed_tokens_per_layer.weight.`
        );
      }
      return null;
    }
    names.push(resolvedName);
  }
  return names;
}

async function loadNamedTensor(ctx, name, label, options = {}) {
  const location = ctx.tensorLocations.get(name) ?? null;
  if (label === 'embedTokensPerLayer') {
    const materializationMode = resolvePerLayerInputMaterializationMode(ctx, label, name, location);
    const effectiveMaterializationMode = (
      options.splitTable === true && materializationMode === 'gpu_split_tables'
    )
      ? 'gpu_resident'
      : materializationMode;
    if (effectiveMaterializationMode === 'range_backed' || effectiveMaterializationMode === 'gpu_split_tables') {
      const rangeBacked = createRangeBackedWeightBuffer(ctx, name, location);
      if (rangeBacked) {
        log.info(
          'Loader',
          `Per-layer input tensor loaded: ${label} <- ${name} ` +
          `(${effectiveMaterializationMode === 'gpu_split_tables' ? 'range-backed CPU source for split GPU tables' : 'range-backed CPU source'})`
        );
        return {
          name,
          tensor: rangeBacked,
        };
      }
      throw new Error(
        `Manifest "${ctx.modelId ?? 'unknown'}" requires range-backed per-layer inputs for ${name}, ` +
        'but shard range loading is unavailable.'
      );
    }
    const toGPU = effectiveMaterializationMode === 'gpu_resident';
    const tensor = await ctx.loadTensor(name, toGPU, true);
    if (!tensor) {
      return null;
    }
    log.info('Loader', `Per-layer input tensor loaded: ${label} <- ${name} (${effectiveMaterializationMode})`);
    return {
      name,
      tensor: wrapRawTensorAsWeightBuffer(ctx, tensor, name),
    };
  }
  const shouldStream = location && typeof ctx.shouldStreamLargeWeight === 'function'
    ? ctx.shouldStreamLargeWeight(name, location, label)
    : false;
  if (label === 'perLayerModelProjection' && shouldStream) {
    const rangeBacked = createRangeBackedWeightBuffer(ctx, name, location);
    if (rangeBacked) {
      log.info(
        'Loader',
        `Per-layer input tensor loaded: ${label} <- ${name} (range-backed CPU source)`
      );
      return {
        name,
        tensor: rangeBacked,
      };
    }
    throw new Error(
      `Manifest "${ctx.modelId ?? 'unknown'}" requires range-backed per-layer input projection for ${name}, ` +
      'but shard range loading is unavailable.'
    );
  }
  let tensor = await ctx.loadTensor(name, !shouldStream, true);
  if (!tensor) {
    return null;
  }
  if (label === 'perLayerModelProjection') {
    const validatedTensor = validateResidentPerLayerProjectionTensor(ctx, name, location, tensor);
    if (validatedTensor !== tensor && isCpuWeightBuffer(validatedTensor)) {
      log.info(
        'Loader',
        `Per-layer input tensor loaded: ${label} <- ${name} (range-backed CPU source)`
      );
      return {
        name,
        tensor: validatedTensor,
      };
    }
    if (validatedTensor !== tensor) {
      tensor = validatedTensor;
    }
  }
  log.info('Loader', `Per-layer input tensor loaded: ${label} <- ${name}`);
  return {
    name,
    tensor: wrapRawTensorAsWeightBuffer(ctx, tensor, name),
  };
}

async function loadOptionalTensor(ctx, candidates, label) {
  for (const name of candidates) {
    const entry = await loadNamedTensor(ctx, name, label);
    if (entry) {
      return entry;
    }
  }
  return null;
}

async function loadSplitEmbedTensors(ctx, numLayers) {
  const tensorNames = resolveSplitPerLayerEmbedTensorNames(ctx.tensorLocations, numLayers);
  if (!tensorNames) {
    return null;
  }
  const entries = [];
  for (const name of tensorNames) {
    const entry = await loadNamedTensor(ctx, name, 'embedTokensPerLayer', {
      splitTable: true,
    });
    if (!entry) {
      return null;
    }
    entries.push(entry);
  }
  return entries;
}

export async function loadPerLayerInputWeights(ctx, architecture) {
  const hiddenSizePerLayerInput = Number(architecture?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }
  const numLayers = Number(architecture?.numLayers ?? 0);

  const splitEmbedEntries = await loadSplitEmbedTensors(ctx, numLayers);
  const [projectionEntry, projectionNormEntry] = await Promise.all([
    loadOptionalTensor(ctx, PROJECTION_TENSOR_CANDIDATES, 'perLayerModelProjection'),
    loadOptionalTensor(ctx, PROJECTION_NORM_TENSOR_CANDIDATES, 'perLayerProjectionNorm'),
  ]);
  const embedEntry = splitEmbedEntries?.[0]
    ?? await loadOptionalTensor(ctx, EMBED_TENSOR_CANDIDATES, 'embedTokensPerLayer');

  if (!embedEntry || !projectionEntry || !projectionNormEntry) {
    const missing = [
      !embedEntry ? 'embed_tokens_per_layer.weight' : null,
      !projectionEntry ? 'per_layer_model_projection.weight' : null,
      !projectionNormEntry ? 'per_layer_projection_norm.weight' : null,
    ].filter(Boolean);
    throw new Error(
      `Manifest "${ctx.modelId ?? 'unknown'}" requires per-layer input weights, ` +
      `but the loader could not resolve: ${missing.join(', ')}.`
    );
  }

  return {
    embedTokensPerLayer: embedEntry.tensor,
    ...(splitEmbedEntries
      ? {
        embedTokensPerLayerSplit: splitEmbedEntries.map((entry) => entry.tensor),
      }
      : {}),
    perLayerModelProjection: projectionEntry.tensor,
    perLayerProjectionNorm: projectionNormEntry.tensor,
  };
}
