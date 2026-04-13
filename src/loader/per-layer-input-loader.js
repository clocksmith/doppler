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
import { acquireBuffer, releaseBuffer, uploadData } from '../memory/buffer-pool.js';
import { QK_K, Q4K_BLOCK_BYTES } from './quantization-constants.js';
import { dequantizeQ4KM, dequantizeQ4KMRowWise } from '../converter/quantizer.js';

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

const PROJECTION_INPUT_ACTIVATION_STATIC_SCALE_CANDIDATES = [
  'model.language_model.per_layer_model_projection.input_activation_static_scale',
  'language_model.per_layer_model_projection.input_activation_static_scale',
  'language_model.model.per_layer_model_projection.input_activation_static_scale',
  'model.per_layer_model_projection.input_activation_static_scale',
  'per_layer_model_projection.input_activation_static_scale',
];

const PROJECTION_OUTPUT_ACTIVATION_STATIC_SCALE_CANDIDATES = [
  'model.language_model.per_layer_model_projection.output_activation_static_scale',
  'language_model.per_layer_model_projection.output_activation_static_scale',
  'language_model.model.per_layer_model_projection.output_activation_static_scale',
  'model.per_layer_model_projection.output_activation_static_scale',
  'per_layer_model_projection.output_activation_static_scale',
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

function isQ4KLocationDtype(dtype) {
  return dtype === 'Q4_K_M' || dtype === 'Q4_K';
}

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return null;
}

async function loadStablePerLayerModelProjection(ctx, name, location) {
  if (!location || !isQ4KLocationDtype(String(location.dtype ?? '').toUpperCase())) {
    return null;
  }
  const shape = Array.isArray(location.shape) ? location.shape : null;
  if (!shape || shape.length !== 2) {
    return null;
  }

  const quantizedBytes = toUint8Array(await ctx.loadTensor(name, false, true));
  if (!quantizedBytes) {
    return null;
  }

  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(cols) || cols <= 0) {
    return null;
  }

  const f32Weights = cols % QK_K === 0
    ? dequantizeQ4KM(quantizedBytes, Math.ceil(location.size / Q4K_BLOCK_BYTES), shape)
    : dequantizeQ4KMRowWise(quantizedBytes, shape);

  const buffer = acquireBuffer(f32Weights.byteLength, undefined, `${name}_f32_reference`);
  try {
    uploadData(buffer, f32Weights);
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
  ctx.gpuBuffers?.add?.(buffer);

  return createWeightBuffer(buffer, 'f32', ctx.resolveWeightLayout(location), shape, name);
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
  if (label === 'perLayerModelProjection') {
    const stabilizedTensor = await loadStablePerLayerModelProjection(ctx, name, location);
    if (stabilizedTensor) {
      log.info(
        'Loader',
        `Per-layer input tensor loaded: ${label} <- ${name} (reference q4k -> f32)`
      );
      return {
        name,
        tensor: stabilizedTensor,
      };
    }
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

function extractScalarTensorValue(data, name, label) {
  if (data instanceof Float32Array) {
    if (data.length !== 1) {
      throw new Error(`${label} "${name}" must resolve to exactly one float32 value.`);
    }
    return data[0];
  }
  if (ArrayBuffer.isView(data)) {
    if (data.byteLength !== Float32Array.BYTES_PER_ELEMENT) {
      throw new Error(`${label} "${name}" must resolve to exactly one float32 value.`);
    }
    return new DataView(data.buffer, data.byteOffset, data.byteLength).getFloat32(0, true);
  }
  if (data instanceof ArrayBuffer) {
    if (data.byteLength !== Float32Array.BYTES_PER_ELEMENT) {
      throw new Error(`${label} "${name}" must resolve to exactly one float32 value.`);
    }
    return new DataView(data).getFloat32(0, true);
  }
  throw new Error(
    `${label} "${name}" must load on CPU as a Float32Array, ArrayBufferView, or ArrayBuffer. ` +
    `Got "${data?.constructor?.name ?? typeof data}".`
  );
}

async function loadNamedScalarTensor(ctx, name, label) {
  const location = ctx.tensorLocations.get(name) ?? null;
  if (!location) {
    return null;
  }
  const shape = Array.isArray(location.shape) ? location.shape : null;
  const elementCount = shape
    ? shape.reduce((total, dimension) => total * Number(dimension), 1)
    : NaN;
  if (!Number.isFinite(elementCount) || elementCount !== 1) {
    throw new Error(
      `${label} "${name}" must have exactly one element. ` +
      `Got shape ${JSON.stringify(location.shape)}.`
    );
  }
  if (String(location.dtype ?? '').toUpperCase() !== 'F32') {
    throw new Error(
      `${label} "${name}" must use dtype F32. ` +
      `Got "${String(location.dtype)}".`
    );
  }
  const tensor = await ctx.loadTensor(name, false, true);
  if (tensor == null) {
    return null;
  }
  const value = extractScalarTensorValue(tensor, name, label);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(
      `${label} "${name}" must be a finite positive float32 scalar. ` +
      `Got ${String(value)}.`
    );
  }
  log.info('Loader', `Per-layer input tensor loaded: ${label} <- ${name} (cpu scalar)`);
  return {
    name,
    value,
  };
}

async function loadOptionalScalarTensor(ctx, candidates, label) {
  for (const name of candidates) {
    const entry = await loadNamedScalarTensor(ctx, name, label);
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
  const [projectionEntry, projectionNormEntry, projectionInputScaleEntry, projectionOutputScaleEntry] = await Promise.all([
    loadOptionalTensor(ctx, PROJECTION_TENSOR_CANDIDATES, 'perLayerModelProjection'),
    loadOptionalTensor(ctx, PROJECTION_NORM_TENSOR_CANDIDATES, 'perLayerProjectionNorm'),
    loadOptionalScalarTensor(
      ctx,
      PROJECTION_INPUT_ACTIVATION_STATIC_SCALE_CANDIDATES,
      'perLayerModelProjectionInputActivationStaticScale'
    ),
    loadOptionalScalarTensor(
      ctx,
      PROJECTION_OUTPUT_ACTIVATION_STATIC_SCALE_CANDIDATES,
      'perLayerModelProjectionOutputActivationStaticScale'
    ),
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
  if ((projectionInputScaleEntry == null) !== (projectionOutputScaleEntry == null)) {
    throw new Error(
      `Manifest "${ctx.modelId ?? 'unknown'}" must resolve both per-layer projection activation static scales together. ` +
      `Got input=${projectionInputScaleEntry?.name ?? 'missing'}, ` +
      `output=${projectionOutputScaleEntry?.name ?? 'missing'}.`
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
    ...(projectionInputScaleEntry && projectionOutputScaleEntry
      ? {
        perLayerModelProjectionInputActivationStaticScale: projectionInputScaleEntry.value,
        perLayerModelProjectionOutputActivationStaticScale: projectionOutputScaleEntry.value,
      }
      : {}),
  };
}
