import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
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

async function loadOptionalTensor(ctx, candidates, label) {
  for (const name of candidates) {
    const location = ctx.tensorLocations.get(name) ?? null;
    if (label === 'embedTokensPerLayer') {
      const materializationMode = resolvePerLayerInputMaterializationMode(ctx, label, name, location);
      if (materializationMode === 'range_backed' || materializationMode === 'gpu_split_tables') {
        const rangeBacked = createRangeBackedWeightBuffer(ctx, name, location);
        if (rangeBacked) {
          log.info(
            'Loader',
            `Per-layer input tensor loaded: ${label} <- ${name} ` +
            `(${materializationMode === 'gpu_split_tables' ? 'range-backed CPU source for split GPU tables' : 'range-backed CPU source'})`
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
      const toGPU = materializationMode === 'gpu_resident';
      const tensor = await ctx.loadTensor(name, toGPU, true);
      if (!tensor) {
        continue;
      }
      log.info('Loader', `Per-layer input tensor loaded: ${label} <- ${name} (${materializationMode})`);
      return {
        name,
        tensor: wrapRawTensorAsWeightBuffer(ctx, tensor, name),
      };
    }
    const shouldStream = location && typeof ctx.shouldStreamLargeWeight === 'function'
      ? ctx.shouldStreamLargeWeight(name, location, label)
      : false;
    const tensor = await ctx.loadTensor(name, !shouldStream, true);
    if (!tensor) {
      continue;
    }
    log.info('Loader', `Per-layer input tensor loaded: ${label} <- ${name}`);
    return {
      name,
      tensor: wrapRawTensorAsWeightBuffer(ctx, tensor, name),
    };
  }
  return null;
}

export async function loadPerLayerInputWeights(ctx, architecture) {
  const hiddenSizePerLayerInput = Number(architecture?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(hiddenSizePerLayerInput) || hiddenSizePerLayerInput <= 0) {
    return null;
  }

  const [embedEntry, projectionEntry, projectionNormEntry] = await Promise.all([
    loadOptionalTensor(ctx, EMBED_TENSOR_CANDIDATES, 'embedTokensPerLayer'),
    loadOptionalTensor(ctx, PROJECTION_TENSOR_CANDIDATES, 'perLayerModelProjection'),
    loadOptionalTensor(ctx, PROJECTION_NORM_TENSOR_CANDIDATES, 'perLayerProjectionNorm'),
  ]);

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
    perLayerModelProjection: projectionEntry.tensor,
    perLayerProjectionNorm: projectionNormEntry.tensor,
  };
}
