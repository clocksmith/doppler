/**
 * Expert Loader - MoE expert weight loading.
 *
 * Handles lazy loading of expert weights for Mixture-of-Experts models.
 * Supports both Mixtral-style (separate tensors) and GPT-OSS-style
 * (packed blocks) expert formats.
 *
 * @module loader/expert-loader
 */

import {
  getShardsForExpert,
  getTensorsForExpert,
  getExpertBytes,
} from '../storage/rdrr-format.js';
import { isWeightBuffer } from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Shard Preloading
// ============================================================================

/**
 * Pre-load specific shards for an expert (lazy loading support).
 *
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx - Expert loader context
 * @param {number} layerIdx - Layer index
 * @param {number} expertIdx - Expert index
 */
export async function preloadShardsForExpert(ctx, layerIdx, expertIdx) {
  // Get required shards from manifest mapping
  const shardIndices = getShardsForExpert(layerIdx, expertIdx);
  if (shardIndices.length === 0) {
    // No mapping available, fall back to loading all shards on demand
    return;
  }

  // Pre-load only the shards needed for this expert
  for (const shardIndex of shardIndices) {
    if (!ctx.shardCache.has(shardIndex)) {
      await ctx.loadShard(shardIndex);
    }
  }
}

// ============================================================================
// Expert Prefetching
// ============================================================================

/**
 * Prefetch experts for next layer (overlap loading with compute).
 * Call this after router selects experts for current layer.
 *
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx - Expert loader context
 * @param {number} nextLayerIdx - Layer index to prefetch for
 * @param {number[]} expertIndices - Expert indices likely to be used
 * @param {boolean} isMoE - Whether model is MoE
 */
export function prefetchExperts(ctx, nextLayerIdx, expertIndices, isMoE) {
  const config = /** @type {import('./loader-types.js').ModelConfig | undefined} */ (ctx.manifest?.config);
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
    await preloadShardsForExpert(ctx, nextLayerIdx, expertIdx);
  });

  // Don't await - let it run in background
  Promise.all(promises).catch((e) => {
    log.warn('Loader', 'Expert prefetch error:', e);
  });
}

/**
 * Get likely experts for next layer based on current layer's routing.
 * Simple heuristic: same experts tend to be selected across layers.
 *
 * @param {number[]} currentExperts - Experts selected in current layer
 * @returns {number[]} Predicted experts for next layer
 */
export function predictNextLayerExperts(currentExperts) {
  // For now, just predict same experts will be used
  // More sophisticated: track expert correlation across layers
  return currentExperts;
}

// ============================================================================
// Expert Loading
// ============================================================================

/**
 * Load expert weights on demand (lazy loading from OPFS).
 *
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx - Expert loader context
 * @param {number} layerIdx - Layer index
 * @param {number} expertIdx - Expert index
 * @returns {Promise<import('./weights.js').ExpertWeights>} Loaded expert weights
 */
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

  // Try Mixtral-style naming first
  let weights = await loadMixtralStyleExpert(ctx, layerIdx, expertIdx);

  // Try GPT-OSS naming if Mixtral naming not found
  if (!weights.gate && !weights.up && !weights.down) {
    weights = await loadGptOssStyleExpert(ctx, layerIdx, expertIdx);
  }

  // Downcast Mixtral-style F32 weights to F16
  if (!weights.isGptOss) {
    await downcastExpertWeights(ctx, weights);
  }

  // Calculate expert size and store in LRU cache
  if (!weights.isGptOss && ctx.expertCache) {
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

/**
 * Load expert using Mixtral-style tensor naming.
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx
 * @param {number} layerIdx
 * @param {number} expertIdx
 * @returns {Promise<import('./weights.js').ExpertWeights>}
 */
async function loadMixtralStyleExpert(ctx, layerIdx, expertIdx) {
  const prefix = `layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;
  const altPrefix = `model.layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;

  return {
    gate: /** @type {GPUBuffer | Float32Array | null} */ (await ctx.loadTensor(`${prefix}.w1.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w1.weight`)),
    up: /** @type {GPUBuffer | Float32Array | null} */ (await ctx.loadTensor(`${prefix}.w3.weight`) ||
        await ctx.loadTensor(`${altPrefix}.w3.weight`)),
    down: /** @type {GPUBuffer | Float32Array | null} */ (await ctx.loadTensor(`${prefix}.w2.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w2.weight`)),
  };
}

/**
 * Load expert using GPT-OSS-style packed tensor naming.
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx
 * @param {number} layerIdx
 * @param {number} expertIdx
 * @returns {Promise<import('./weights.js').ExpertWeights>}
 */
async function loadGptOssStyleExpert(ctx, layerIdx, expertIdx) {
  const gptOssPrefix = `model.layers.${layerIdx}.mlp.experts`;
  const packedKey = `layer_${layerIdx}_gptoss_packed`;
  let packed = ctx.experts.get(packedKey);

  if (!packed) {
    const config = /** @type {import('./loader-types.js').ModelConfig | undefined} */ (ctx.manifest?.config);
    const numExpertsFromConfig = config?.num_local_experts || config?.num_experts || 32;

    packed = {
      isGptOss: true,
      numExperts: numExpertsFromConfig,
      gateUpBlocks: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_blocks`)),
      gateUpScales: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_scales`)),
      gateUpBias: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_bias`)),
      downBlocks: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.down_proj_blocks`)),
      downScales: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.down_proj_scales`)),
      downBias: /** @type {GPUBuffer | null} */ (await ctx.loadTensor(`${gptOssPrefix}.down_proj_bias`)),
    };

    ctx.experts.set(packedKey, packed);
  }

  return {
    isGptOss: true,
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

/**
 * Downcast expert weights from F32 to F16.
 * @param {import('./expert-loader.js').ExpertLoaderContext} ctx
 * @param {import('./weights.js').ExpertWeights} weights
 */
async function downcastExpertWeights(ctx, weights) {
  for (const k of /** @type {const} */ (['gate', 'up', 'down'])) {
    const buf = weights[k];
    if (!buf) continue;

    // Only downcast GPUBuffer or WeightBuffer (not Float32Array)
    if (!(buf instanceof GPUBuffer) && !isWeightBuffer(buf)) {
      continue;
    }

    const result = await maybeDowncastToF16(/** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer} */ (buf), {
      label: `expert_${k}`,
      keepF32: ctx.keepF32Weights,
    });

    if (result?.wasDowncast) {
      weights[k] = /** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null} */ (result.buffer);
      if (result.newBuffer) {
        ctx.gpuBuffers.add(result.newBuffer);
      }
    }
  }
}

/**
 * Calculate total size of expert weights in bytes.
 * @param {import('./weights.js').ExpertWeights} weights
 * @returns {number}
 */
function calculateExpertSize(weights) {
  let sizeBytes = 0;

  for (const k of /** @type {const} */ (['gate', 'up', 'down'])) {
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
