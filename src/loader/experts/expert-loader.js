

import {
  getShardsForExpert,
  getTensorsForExpert,
  getExpertBytes,
} from '../../formats/rdrr/index.js';
import { isWeightBuffer } from '../../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from '../weight-downcast.js';
import { log, trace as debugTrace } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { isBufferActive, releaseBuffer } from '../../memory/buffer-pool.js';

// ============================================================================
// Shard Preloading
// ============================================================================


export async function preloadShardsForExpert(ctx, layerIdx, expertIdx, options) {
  const packedShardIndices = getPackedExpertShardIndices(ctx, layerIdx);
  if (packedShardIndices.length > 0) {
    for (const shardIndex of packedShardIndices) {
      if (!ctx.shardCache.has(shardIndex)) {
        await ctx.loadShard(shardIndex, options);
      }
    }
    return;
  }

  if (isPackedExpertFormat(ctx.manifest?.moeConfig?.expertFormat)) {
    return;
  }

  // Get required shards from manifest mapping
  const shardIndices = getShardsForExpert(layerIdx, expertIdx, ctx.manifest);
  if (shardIndices.length === 0) {
    // No mapping available, fall back to loading all shards on demand
    return;
  }

  // Pre-load only the shards needed for this expert
  for (const shardIndex of shardIndices) {
    if (!ctx.shardCache.has(shardIndex)) {
      await ctx.loadShard(shardIndex, options);
    }
  }
}

// ============================================================================
// Expert Prefetching
// ============================================================================


export function prefetchExperts(ctx, nextLayerIdx, expertIndices, isMoE) {
  const config =  (ctx.manifest?.config);
  const numLayers = config?.num_hidden_layers ?? 0;

  if (!isMoE || nextLayerIdx >= numLayers) {
    return;
  }

  // Fire-and-forget: load shards in background
  // This overlaps shard loading with current layer's compute
  const promises = expertIndices.map(async (expertIdx) => {
    // Check if already cached
    if (ctx.expertCache?.has(nextLayerIdx, expertIdx)) {
      return;
    }
    // Pre-load the shards (not the full expert tensor upload)
    await preloadShardsForExpert(ctx, nextLayerIdx, expertIdx, { priority: 'low' });
  });

  // Don't await - let it run in background
  Promise.all(promises).catch((e) => {
    log.warn('Loader', 'Expert prefetch error:', e);
  });
}


export function predictNextLayerExperts(currentExperts) {
  // For now, just predict same experts will be used
  // More sophisticated: track expert correlation across layers
  return currentExperts;
}

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

// ============================================================================
// Expert Loading
// ============================================================================


export async function loadExpert(ctx, layerIdx, expertIdx) {
  // Check LRU cache first
  if (ctx.expertCache) {
    const cached = ctx.expertCache.get(layerIdx, expertIdx);
    if (cached) {
      return cached;
    }
  }

  // Fall back to simple map for non-cached packed experts.
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  if (ctx.experts.has(key)) {
    return ctx.experts.get(key);
  }

  debugTrace.loader(`Loading expert ${expertIdx} for layer ${layerIdx}`);

  // Pre-load only the shards containing this expert's tensors
  await preloadShardsForExpert(ctx, layerIdx, expertIdx);

  // Get tensor names from manifest if available (for logging/debugging)
  const expertFormat = resolveExpertFormat(ctx);
  const tensorNames = getExpertTensorNames(ctx, layerIdx, expertIdx, expertFormat);
  if (tensorNames.length > 0) {
    debugTrace.loader(`Expert ${layerIdx}_${expertIdx} tensors: ${tensorNames.length}`);
  }

  let weights;
  if (expertFormat === 'gpt-oss') {
    weights = await loadGptOssStyleExpert(ctx, layerIdx, expertIdx);
    assertGptOssWeights(weights, layerIdx, expertIdx);
  } else if (expertFormat === 'gemma4') {
    weights = await loadGemma4StyleExpert(ctx, layerIdx, expertIdx);
    assertGemma4Weights(weights, layerIdx, expertIdx);
  } else {
    weights = await loadMixtralStyleExpert(ctx, layerIdx, expertIdx);
    assertMixtralWeights(weights, layerIdx, expertIdx);
  }

  // Downcast Mixtral-style F32 weights to F16
  weights.expertFormat = expertFormat;
  if (expertFormat === 'mixtral') {
    await downcastExpertWeights(ctx, weights);
  }

  // Calculate expert size and store in LRU cache
  if (expertFormat === 'mixtral' && ctx.expertCache) {
    const sizeBytes = calculateExpertSize(weights);
    ctx.expertCache.put(layerIdx, expertIdx, weights, sizeBytes);
  } else {
    // Packed expert formats use the simple map (shared across experts).
    ctx.experts.set(key, weights);
  }

  return weights;
}

// ============================================================================
// Internal Helpers
// ============================================================================


async function loadMixtralStyleExpert(ctx, layerIdx, expertIdx) {
  const prefix = `layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;
  const altPrefix = `model.layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;

  return {
    gate:  (await ctx.loadTensor(`${prefix}.w1.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w1.weight`)),
    up:  (await ctx.loadTensor(`${prefix}.w3.weight`) ||
        await ctx.loadTensor(`${altPrefix}.w3.weight`)),
    down:  (await ctx.loadTensor(`${prefix}.w2.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w2.weight`)),
  };
}

function isPackedExpertFormat(expertFormat) {
  return expertFormat === 'gpt-oss' || expertFormat === 'gemma4';
}

function resolveExpertFormat(ctx) {
  const manifest = ctx.manifest ?? {};
  const moeConfig = manifest.moeConfig ?? null;
  const modelId = manifest.modelId ?? 'unknown';
  if (!moeConfig) {
    throw new Error(
      `[MoE] Manifest "${modelId}" missing moeConfig. ` +
      'Re-convert the model using the latest converter.'
    );
  }

  const expertFormat = moeConfig.expertFormat;
  if (expertFormat === 'gpt-oss' || expertFormat === 'mixtral' || expertFormat === 'gemma4') {
    return expertFormat;
  }
  if (expertFormat == null) {
    throw new Error(
      `[MoE] Manifest "${modelId}" missing moeConfig.expertFormat. ` +
      'Re-convert the model using the latest converter.'
    );
  }
  throw new Error(`[MoE] Manifest "${modelId}" has invalid expertFormat "${expertFormat}".`);
}

function getShardIndicesForLocation(location) {
  if (!location) return [];
  if (Array.isArray(location.spans) && location.spans.length > 0) {
    return Array.from(new Set(location.spans.map((span) => span?.shardIndex).filter(Number.isInteger)));
  }
  return Number.isInteger(location.shardIndex) ? [location.shardIndex] : [];
}

function getExistingTensorNames(ctx, candidates) {
  const tensorLocations = ctx.tensorLocations;
  if (!tensorLocations || typeof tensorLocations.get !== 'function') {
    return [];
  }
  const names = [];
  for (const candidateGroup of candidates) {
    const found = candidateGroup.find((name) => tensorLocations.has(name));
    if (found) {
      names.push(found);
    }
  }
  return names;
}

function getGemma4ExpertTensorCandidates(layerIdx) {
  return [
    [
      `model.decoder.layers.${layerIdx}.experts.gate_up_proj`,
      `model.encoder.language_model.layers.${layerIdx}.experts.gate_up_proj`,
      `model.layers.${layerIdx}.experts.gate_up_proj`,
    ],
    [
      `model.decoder.layers.${layerIdx}.experts.down_proj`,
      `model.encoder.language_model.layers.${layerIdx}.experts.down_proj`,
      `model.layers.${layerIdx}.experts.down_proj`,
    ],
  ];
}

function getGptOssExpertTensorCandidates(layerIdx) {
  const prefix = `model.layers.${layerIdx}.mlp.experts`;
  return [
    [`${prefix}.gate_up_proj_blocks`],
    [`${prefix}.gate_up_proj_scales`],
    [`${prefix}.gate_up_proj_bias`],
    [`${prefix}.down_proj_blocks`],
    [`${prefix}.down_proj_scales`],
    [`${prefix}.down_proj_bias`],
  ];
}

function getPackedExpertTensorNames(ctx, layerIdx, expertFormat = ctx.manifest?.moeConfig?.expertFormat) {
  if (expertFormat === 'gemma4') {
    return getExistingTensorNames(ctx, getGemma4ExpertTensorCandidates(layerIdx));
  }
  if (expertFormat === 'gpt-oss') {
    return getExistingTensorNames(ctx, getGptOssExpertTensorCandidates(layerIdx));
  }
  return [];
}

function getPackedExpertShardIndices(ctx, layerIdx) {
  const tensorNames = getPackedExpertTensorNames(ctx, layerIdx);
  const shardIndices = new Set();
  for (const name of tensorNames) {
    const location = ctx.tensorLocations?.get?.(name);
    for (const shardIndex of getShardIndicesForLocation(location)) {
      shardIndices.add(shardIndex);
    }
  }
  return Array.from(shardIndices);
}

function getExpertTensorNames(ctx, layerIdx, expertIdx, expertFormat) {
  const packedTensorNames = getPackedExpertTensorNames(ctx, layerIdx, expertFormat);
  if (packedTensorNames.length > 0) {
    return packedTensorNames;
  }
  if (isPackedExpertFormat(expertFormat) && !ctx.manifest?.groups?.[`layer.${layerIdx}.expert.${expertIdx}`]) {
    return [];
  }
  return getTensorsForExpert(layerIdx, expertIdx, ctx.manifest);
}

function resolveGptOssNumExperts(ctx) {
  const manifest = ctx.manifest ?? {};
  const numExperts = manifest.moeConfig?.numExperts ?? null;

  if (numExperts == null) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(`[MoE] GPT-OSS manifest "${modelId}" missing moeConfig.numExperts`);
  }

  return numExperts;
}

function resolveGemma4NumExperts(ctx) {
  const manifest = ctx.manifest ?? {};
  const numExperts = manifest.moeConfig?.numExperts ?? null;
  if (numExperts == null) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(`[MoE] Gemma-style manifest "${modelId}" missing moeConfig.numExperts`);
  }
  return numExperts;
}

function resolveGemma4ExpertIntermediateSize(ctx) {
  const manifest = ctx.manifest ?? {};
  const expertIntermediateSize = manifest.moeConfig?.expertIntermediateSize ?? null;
  if (expertIntermediateSize == null) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(`[MoE] Gemma-style manifest "${modelId}" missing moeConfig.expertIntermediateSize`);
  }
  return expertIntermediateSize;
}

async function loadFirstTensor(ctx, names) {
  for (const name of names) {
    const tensor = await ctx.loadTensor(name);
    if (tensor) return tensor;
  }
  return null;
}

function assertMixtralWeights(weights, layerIdx, expertIdx) {
  const missing = [];
  if (!weights.gate) missing.push('gate');
  if (!weights.up) missing.push('up');
  if (!weights.down) missing.push('down');
  if (missing.length > 0) {
    throw new Error(
      `[MoE] Expert ${layerIdx}_${expertIdx} missing tensors: ${missing.join(', ')}`
    );
  }
}

function assertGptOssWeights(weights, layerIdx, expertIdx) {
  const missing = [];
  if (!weights.gateUpBlocks) missing.push('gate_up_proj_blocks');
  if (!weights.gateUpScales) missing.push('gate_up_proj_scales');
  if (!weights.gateUpBias) missing.push('gate_up_proj_bias');
  if (!weights.downBlocks) missing.push('down_proj_blocks');
  if (!weights.downScales) missing.push('down_proj_scales');
  if (missing.length > 0) {
    throw new Error(
      `[MoE] GPT-OSS expert ${layerIdx}_${expertIdx} missing tensors: ${missing.join(', ')}`
    );
  }
}

function assertGemma4Weights(weights, layerIdx, expertIdx) {
  const missing = [];
  if (!weights.gateUp) missing.push('experts.gate_up_proj');
  if (!weights.down) missing.push('experts.down_proj');
  if (!weights.expertIntermediateSize) missing.push('expertIntermediateSize');
  if (missing.length > 0) {
    throw new Error(
      `[MoE] Gemma-style expert ${layerIdx}_${expertIdx} missing tensors: ${missing.join(', ')}`
    );
  }
}

async function loadGemma4StyleExpert(ctx, layerIdx, expertIdx) {
  const packedKey = `layer_${layerIdx}_gemma4_packed`;
  let packed = ctx.experts.get(packedKey);

  if (!packed) {
    packed = {
      expertFormat: 'gemma4',
      numExperts: resolveGemma4NumExperts(ctx),
      expertIntermediateSize: resolveGemma4ExpertIntermediateSize(ctx),
      gateUp: await loadFirstTensor(ctx, getGemma4ExpertTensorCandidates(layerIdx)[0]),
      down: await loadFirstTensor(ctx, getGemma4ExpertTensorCandidates(layerIdx)[1]),
    };
    ctx.experts.set(packedKey, packed);
  }

  return {
    expertFormat: 'gemma4',
    expertIdx,
    numExperts: packed.numExperts,
    expertIntermediateSize: packed.expertIntermediateSize,
    gateUp: packed.gateUp,
    down: packed.down,
  };
}


async function loadGptOssStyleExpert(ctx, layerIdx, expertIdx) {
  const gptOssPrefix = `model.layers.${layerIdx}.mlp.experts`;
  const packedKey = `layer_${layerIdx}_gptoss_packed`;
  evictStaleGptOssLayers(ctx, layerIdx);
  let packed = ctx.experts.get(packedKey);

  if (!packed) {
    const numExpertsFromConfig = resolveGptOssNumExperts(ctx);

    packed = {
      expertFormat: 'gpt-oss',
      numExperts: numExpertsFromConfig,
      gateUpBlocks:  (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_blocks`)),
      gateUpScales:  (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_scales`)),
      gateUpBias:  (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_bias`)),
      downBlocks:  (await ctx.loadTensor(`${gptOssPrefix}.down_proj_blocks`)),
      downScales:  (await ctx.loadTensor(`${gptOssPrefix}.down_proj_scales`)),
      downBias:  (await ctx.loadTensor(`${gptOssPrefix}.down_proj_bias`)),
    };

    ctx.experts.set(packedKey, packed);
    touchGptOssLayer(ctx, layerIdx);
  } else {
    touchGptOssLayer(ctx, layerIdx);
  }

  return {
    expertFormat: 'gpt-oss',
    expertIdx,
    numExperts: packed.numExperts,
    gateUpBlocks: packed.gateUpBlocks,
    gateUpScales: packed.gateUpScales,
    gateUpBias: packed.gateUpBias,
    downBlocks: packed.downBlocks,
    downScales: packed.downScales,
    downBias: packed.downBias,
  };
}

function touchGptOssLayer(ctx, layerIdx) {
  if (!ctx.gptOssLayerAccess) {
    ctx.gptOssLayerAccess = new Map();
  }
  ctx.gptOssLayerAccess.set(layerIdx, Date.now());
}

function getGpuBuffer(value) {
  if (isWeightBuffer(value)) {
    return value.buffer;
  }
  if (isGpuBufferInstance(value)) {
    return value;
  }
  return null;
}

function releasePackedLayerWeights(ctx, packed) {
  const buffers = [
    packed.gateUpBlocks,
    packed.gateUpScales,
    packed.gateUpBias,
    packed.downBlocks,
    packed.downScales,
    packed.downBias,
  ];
  for (const entry of buffers) {
    const gpuBuffer = getGpuBuffer(entry);
    if (!gpuBuffer) continue;
    try {
      if (isBufferActive(gpuBuffer)) {
        releaseBuffer(gpuBuffer);
      } else {
        gpuBuffer.destroy();
      }
      ctx.gpuBuffers?.delete?.(gpuBuffer);
    } catch {
      // Ignore already-released buffers.
    }
  }
}

function evictStaleGptOssLayers(ctx, activeLayerIdx) {
  const runtime = getRuntimeConfig();
  const pager = runtime.loading.expertCache.gptOssPager ?? {
    enabled: true,
    maxResidentLayers: 2,
  };

  if (!pager.enabled || !Number.isFinite(pager.maxResidentLayers) || pager.maxResidentLayers <= 0) {
    return;
  }

  const entries = [];
  for (const key of ctx.experts.keys()) {
    const match = /^layer_(\d+)_gptoss_packed$/.exec(key);
    if (!match) continue;
    const layerIdx = Number(match[1]);
    const lastAccess = ctx.gptOssLayerAccess?.get(layerIdx) ?? 0;
    entries.push({ key, layerIdx, lastAccess });
  }

  const hasActive = entries.some((entry) => entry.layerIdx === activeLayerIdx);
  const maxAllowed = hasActive ? pager.maxResidentLayers : pager.maxResidentLayers - 1;
  if (entries.length <= maxAllowed) {
    return;
  }

  entries.sort((a, b) => {
    if (a.layerIdx === activeLayerIdx) return 1;
    if (b.layerIdx === activeLayerIdx) return -1;
    return a.lastAccess - b.lastAccess;
  });

  while (entries.length > maxAllowed) {
    const evicted = entries.shift();
    if (!evicted || evicted.layerIdx === activeLayerIdx) {
      break;
    }
    const packed = ctx.experts.get(evicted.key);
    if (!packed) {
      continue;
    }
    releasePackedLayerWeights(ctx, packed);
    ctx.experts.delete(evicted.key);
    ctx.gptOssLayerAccess?.delete(evicted.layerIdx);
    log.info('Loader', `Evicted GPT-OSS packed experts for layer ${evicted.layerIdx}`);
  }
}


async function downcastExpertWeights(ctx, weights) {
  for (const k of  (['gate', 'up', 'down'])) {
    const buf = weights[k];
    if (!buf) continue;

    // Only downcast GPUBuffer or WeightBuffer (not Float32Array)
    if (!isGpuBufferInstance(buf) && !isWeightBuffer(buf)) {
      continue;
    }

    const result = await maybeDowncastToF16( (buf), {
      label: `expert_${k}`,
      keepF32: ctx.keepF32Weights,
      dtype: isWeightBuffer(buf) ? buf.dtype : null,
    });

    if (result?.wasDowncast) {
      weights[k] =  (result.buffer);
      if (result.newBuffer) {
        ctx.gpuBuffers.add(result.newBuffer);
      }
    }
  }
}


function calculateExpertSize(weights) {
  let sizeBytes = 0;

  for (const k of  (['gate', 'up', 'down'])) {
    const buf = weights[k];
    if (isWeightBuffer(buf)) {
      sizeBytes += buf.buffer.size;
    } else if (isGpuBufferInstance(buf)) {
      sizeBytes += buf.size;
    }
  }

  // Use manifest-provided expert size if available, otherwise use calculated
  const manifestBytes = getExpertBytes(ctx.manifest);
  if (manifestBytes > 0) {
    sizeBytes = manifestBytes;
  }

  return sizeBytes;
}
