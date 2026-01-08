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
  type RDRRManifest,
} from '../storage/rdrr-format.js';
import { ExpertCache } from './expert-cache.js';
import type { ExpertWeights } from './weights.js';
import type { ModelConfig } from './loader-types.js';
import { isWeightBuffer, type WeightBuffer } from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Types
// ============================================================================

/** Tensor loading function signature */
export type TensorLoader = (
  name: string,
  toGPU?: boolean,
  silent?: boolean
) => Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null>;

/** Shard loading function signature */
export type ShardLoader = (shardIndex: number) => Promise<ArrayBuffer>;

/** Shard cache interface */
export interface ShardCacheInterface {
  has(shardIndex: number): boolean;
}

/**
 * Context required for expert loading operations.
 */
export interface ExpertLoaderContext {
  /** Model manifest */
  manifest: RDRRManifest | null;
  /** Load a tensor by name */
  loadTensor: TensorLoader;
  /** Load a shard by index */
  loadShard: ShardLoader;
  /** Shard cache for checking loaded shards */
  shardCache: ShardCacheInterface;
  /** Expert LRU cache */
  expertCache: ExpertCache | null;
  /** Simple map for GPT-OSS packed experts */
  experts: Map<string, ExpertWeights>;
  /** GPU buffers to track for cleanup */
  gpuBuffers: Set<GPUBuffer>;
  /** Keep F32 weights (skip downcast) */
  keepF32Weights: boolean;
}

// ============================================================================
// Shard Preloading
// ============================================================================

/**
 * Pre-load specific shards for an expert (lazy loading support).
 *
 * @param ctx - Expert loader context
 * @param layerIdx - Layer index
 * @param expertIdx - Expert index
 */
export async function preloadShardsForExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<void> {
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
 * @param ctx - Expert loader context
 * @param nextLayerIdx - Layer index to prefetch for
 * @param expertIndices - Expert indices likely to be used
 * @param isMoE - Whether model is MoE
 */
export function prefetchExperts(
  ctx: ExpertLoaderContext,
  nextLayerIdx: number,
  expertIndices: number[],
  isMoE: boolean
): void {
  const config = ctx.manifest?.config as ModelConfig | undefined;
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
 * @param currentExperts - Experts selected in current layer
 * @returns Predicted experts for next layer
 */
export function predictNextLayerExperts(currentExperts: number[]): number[] {
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
 * @param ctx - Expert loader context
 * @param layerIdx - Layer index
 * @param expertIdx - Expert index
 * @returns Loaded expert weights
 */
export async function loadExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<ExpertWeights> {
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
    return ctx.experts.get(key)!;
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
 */
async function loadMixtralStyleExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<ExpertWeights> {
  const prefix = `layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;
  const altPrefix = `model.layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;

  return {
    gate: (await ctx.loadTensor(`${prefix}.w1.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w1.weight`)) as GPUBuffer | Float32Array | null,
    up: (await ctx.loadTensor(`${prefix}.w3.weight`) ||
        await ctx.loadTensor(`${altPrefix}.w3.weight`)) as GPUBuffer | Float32Array | null,
    down: (await ctx.loadTensor(`${prefix}.w2.weight`) ||
          await ctx.loadTensor(`${altPrefix}.w2.weight`)) as GPUBuffer | Float32Array | null,
  };
}

/**
 * Load expert using GPT-OSS-style packed tensor naming.
 */
async function loadGptOssStyleExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<ExpertWeights> {
  const gptOssPrefix = `model.layers.${layerIdx}.mlp.experts`;
  const packedKey = `layer_${layerIdx}_gptoss_packed`;
  let packed = ctx.experts.get(packedKey);

  if (!packed) {
    const config = ctx.manifest?.config as ModelConfig | undefined;
    const numExpertsFromConfig = config?.num_local_experts || config?.num_experts || 32;

    packed = {
      isGptOss: true,
      numExperts: numExpertsFromConfig,
      gateUpBlocks: await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_blocks`) as GPUBuffer | null,
      gateUpScales: await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_scales`) as GPUBuffer | null,
      gateUpBias: await ctx.loadTensor(`${gptOssPrefix}.gate_up_proj_bias`) as GPUBuffer | null,
      downBlocks: await ctx.loadTensor(`${gptOssPrefix}.down_proj_blocks`) as GPUBuffer | null,
      downScales: await ctx.loadTensor(`${gptOssPrefix}.down_proj_scales`) as GPUBuffer | null,
      downBias: await ctx.loadTensor(`${gptOssPrefix}.down_proj_bias`) as GPUBuffer | null,
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
 */
async function downcastExpertWeights(
  ctx: ExpertLoaderContext,
  weights: ExpertWeights
): Promise<void> {
  for (const k of ['gate', 'up', 'down'] as const) {
    const buf = weights[k];
    if (!buf) continue;

    // Only downcast GPUBuffer or WeightBuffer (not Float32Array)
    if (!(buf instanceof GPUBuffer) && !isWeightBuffer(buf)) {
      continue;
    }

    const result = await maybeDowncastToF16(buf as GPUBuffer | WeightBuffer, {
      label: `expert_${k}`,
      keepF32: ctx.keepF32Weights,
    });

    if (result?.wasDowncast) {
      weights[k] = result.buffer as GPUBuffer | WeightBuffer | Float32Array | null;
      if (result.newBuffer) {
        ctx.gpuBuffers.add(result.newBuffer);
      }
    }
  }
}

/**
 * Calculate total size of expert weights in bytes.
 */
function calculateExpertSize(weights: ExpertWeights): number {
  let sizeBytes = 0;

  for (const k of ['gate', 'up', 'down'] as const) {
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
