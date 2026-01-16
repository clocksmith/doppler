

import {
  getShardsForExpert,
  getTensorsForExpert,
  getExpertBytes,
} from '../storage/rdrr-format.js';
import { isWeightBuffer } from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { releaseBuffer } from '../memory/buffer-pool.js';

// ============================================================================
// Shard Preloading
// ============================================================================


export async function preloadShardsForExpert(ctx, layerIdx, expertIdx, options) {
  // Get required shards from manifest mapping
  const shardIndices = getShardsForExpert(layerIdx, expertIdx);
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

  // Fall back to simple map for non-cached experts (GPT-OSS packed weights)
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  if (ctx.experts.has(key)) {
    return ctx.experts.get(key);
  }

  debugTrace.loader(`Loading expert ${expertIdx} for layer ${layerIdx}`);

  // Pre-load only the shards containing this expert's tensors
  await preloadShardsForExpert(ctx, layerIdx, expertIdx);

  // Get tensor names from manifest if available (for logging/debugging)
  const tensorNames = getTensorsForExpert(layerIdx, expertIdx);
  if (tensorNames.length > 0) {
    debugTrace.loader(`Expert ${layerIdx}_${expertIdx} tensors: ${tensorNames.length}`);
  }

  const expertFormat = resolveExpertFormat(ctx);
  let weights;
  if (expertFormat === 'gpt-oss') {
    weights = await loadGptOssStyleExpert(ctx, layerIdx, expertIdx);
    assertGptOssWeights(weights, layerIdx, expertIdx);
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
    // GPT-OSS packed weights use the simple map (shared across experts)
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
  if (expertFormat === 'gpt-oss' || expertFormat === 'mixtral') {
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

function resolveGptOssNumExperts(ctx) {
  const manifest = ctx.manifest ?? {};
  const numExperts = manifest.moeConfig?.numExperts ?? null;

  if (numExperts == null) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(`[MoE] GPT-OSS manifest "${modelId}" missing moeConfig.numExperts`);
  }

  return numExperts;
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
  if (value instanceof GPUBuffer) {
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
      releaseBuffer(gpuBuffer);
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
    if (!(buf instanceof GPUBuffer) && !isWeightBuffer(buf)) {
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
    } else if (buf instanceof GPUBuffer) {
      sizeBytes += buf.size;
    }
  }

  // Use manifest-provided expert size if available, otherwise use calculated
  const manifestBytes = getExpertBytes();
  if (manifestBytes > 0) {
    sizeBytes = manifestBytes;
  }

  return sizeBytes;
}
