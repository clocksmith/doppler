/**
 * Expert Loader - MoE expert weight loading.
 *
 * Handles lazy loading of expert weights for Mixture-of-Experts models.
 * Supports both Mixtral-style (separate tensors) and GPT-OSS-style
 * (packed blocks) expert formats.
 *
 * @module loader/expert-loader
 */

import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { ExpertCache } from './expert-cache.js';
import type { ExpertWeights } from './weights.js';
import type { WeightBuffer } from '../gpu/weight-buffer.js';

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

/**
 * Pre-load specific shards for an expert (lazy loading support).
 *
 * @param ctx - Expert loader context
 * @param layerIdx - Layer index
 * @param expertIdx - Expert index
 */
export declare function preloadShardsForExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<void>;

/**
 * Prefetch experts for next layer (overlap loading with compute).
 * Call this after router selects experts for current layer.
 *
 * @param ctx - Expert loader context
 * @param nextLayerIdx - Layer index to prefetch for
 * @param expertIndices - Expert indices likely to be used
 * @param isMoE - Whether model is MoE
 */
export declare function prefetchExperts(
  ctx: ExpertLoaderContext,
  nextLayerIdx: number,
  expertIndices: number[],
  isMoE: boolean
): void;

/**
 * Get likely experts for next layer based on current layer's routing.
 * Simple heuristic: same experts tend to be selected across layers.
 *
 * @param currentExperts - Experts selected in current layer
 * @returns Predicted experts for next layer
 */
export declare function predictNextLayerExperts(currentExperts: number[]): number[];

/**
 * Load expert weights on demand (lazy loading from OPFS).
 *
 * @param ctx - Expert loader context
 * @param layerIdx - Layer index
 * @param expertIdx - Expert index
 * @returns Loaded expert weights
 */
export declare function loadExpert(
  ctx: ExpertLoaderContext,
  layerIdx: number,
  expertIdx: number
): Promise<ExpertWeights>;
