/**
 * Loader Types
 *
 * Type definitions for the DopplerLoader.
 *
 * @module loader/loader-types
 */

import type { CpuWeightBuffer, WeightBuffer } from '../gpu/weight-buffer.js';
import type { TensorRole } from '../config/schema/index.js';

/**
 * Tensor location in loaded model
 */
export interface TensorLocation {
  shardIndex: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
  group?: string;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  /** Weight storage layout: 'column' means pre-transposed for faster matmul */
  layout?: 'row' | 'column';
  /** Original shape before transpose (if layout is 'column') */
  originalShape?: number[];
}

/**
 * Loaded layer weights
 */
export interface LayerWeights {
  inputNorm: GPUBuffer | Float32Array | null;
  qProj: GPUBuffer | WeightBuffer | Float32Array | null;
  kProj: GPUBuffer | WeightBuffer | Float32Array | null;
  vProj: GPUBuffer | WeightBuffer | Float32Array | null;
  oProj: GPUBuffer | WeightBuffer | Float32Array | null;
  qkvProj?: GPUBuffer | WeightBuffer | Float32Array | null;
  qkvSizes?: [number, number, number] | null;
  qkvDtype?: 'f16' | 'f32' | null;
  linearInProjZ?: GPUBuffer | WeightBuffer | Float32Array | null;
  linearInProjA?: GPUBuffer | WeightBuffer | Float32Array | null;
  linearInProjB?: GPUBuffer | WeightBuffer | Float32Array | null;
  linearConv1D?: GPUBuffer | Float32Array | null;
  linearDtBias?: GPUBuffer | Float32Array | null;
  linearALog?: GPUBuffer | Float32Array | null;
  linearNorm?: GPUBuffer | Float32Array | null;
  qNorm: GPUBuffer | Float32Array | null;
  kNorm: GPUBuffer | Float32Array | null;
  postAttentionNorm: GPUBuffer | Float32Array | null;
  preFeedforwardNorm: GPUBuffer | Float32Array | null;
  postFeedforwardNorm: GPUBuffer | Float32Array | null;
  postNorm: GPUBuffer | Float32Array | null;
  postAttnNorm: GPUBuffer | Float32Array | null;
  convInProj?: GPUBuffer | WeightBuffer | Float32Array | null;
  convKernel?: GPUBuffer | WeightBuffer | Float32Array | null;
  convOutProj?: GPUBuffer | WeightBuffer | Float32Array | null;
  ffnGate: GPUBuffer | WeightBuffer | Float32Array | null;
  ffnUp: GPUBuffer | WeightBuffer | Float32Array | null;
  ffnDown: GPUBuffer | WeightBuffer | Float32Array | null;
  /** Fused gate+up projection [intermediateSize*2, hiddenSize] for 2-pass FFN */
  ffnGateUp?: GPUBuffer | WeightBuffer | Float32Array | null;
  // Aliases for pipeline compatibility
  gate?: GPUBuffer | WeightBuffer | Float32Array | null;
  up?: GPUBuffer | WeightBuffer | Float32Array | null;
  down?: GPUBuffer | WeightBuffer | Float32Array | null;
  /** Fused gate+up for pipeline compatibility */
  gateUp?: GPUBuffer | WeightBuffer | Float32Array | null;
  routerWeight?: GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null;
  routerBias?: GPUBuffer | Float32Array | null;
  attentionSinks?: GPUBuffer | Float32Array | null;
  perLayerInputGate?: GPUBuffer | WeightBuffer | Float32Array | null;
  perLayerProjection?: GPUBuffer | WeightBuffer | Float32Array | null;
  postPerLayerInputNorm?: GPUBuffer | Float32Array | null;
  layerScalar?: GPUBuffer | Float32Array | null;
}

export interface PerLayerInputWeights {
  embedTokensPerLayer: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  perLayerModelProjection: GPUBuffer | WeightBuffer | Float32Array | null;
  perLayerProjectionNorm: GPUBuffer | Float32Array | null;
}

/**
 * Loading progress information
 */
export interface LoadProgress {
  stage: 'manifest' | 'shards' | 'layers' | 'gpu_transfer' | 'complete';
  progress: number;
  /** Current layer index */
  layer?: number;
  /** Total layers */
  total?: number;
  /** Current shard index */
  shard?: number;
  /** Total shards */
  totalShards?: number;
  /** Bytes loaded so far */
  bytesLoaded?: number;
  /** Total bytes to load */
  totalBytes?: number;
  /** Loading speed in bytes per second */
  bytesPerSecond?: number;
  /** Human-readable message */
  message?: string;
}

/**
 * Loading options
 */
export interface LoadOptions {
  onProgress?: (progress: LoadProgress) => void;
  verifyHashes: boolean;
}

/**
 * Shard load priority.
 */
export type ShardLoadPriority = 'high' | 'low';

/**
 * Shard loading options.
 */
export interface ShardLoadOptions {
  priority?: ShardLoadPriority;
}

/**
 * Custom shard loader options
 */
export interface CustomShardLoaderOptions {
  verify?: boolean;
  loadShardRange?: CustomShardRangeLoader;
  streamShardRange?: CustomShardStreamLoader;
}

/**
 * Custom shard loader function
 */
export type CustomShardLoader = (
  shardIndex: number
) => Promise<ArrayBuffer | Uint8Array>;

/**
 * Custom shard range loader function
 */
export type CustomShardRangeLoader = (
  shardIndex: number,
  offset: number,
  length?: number | null
) => Promise<ArrayBuffer | Uint8Array>;

/**
 * Custom shard range stream options
 */
export interface CustomShardStreamOptions {
  chunkBytes?: number;
}

/**
 * Custom shard range stream loader function
 */
export type CustomShardStreamLoader = (
  shardIndex: number,
  offset?: number,
  length?: number | null,
  options?: CustomShardStreamOptions
) => AsyncIterable<Uint8Array>;

/**
 * Loader statistics
 */
export interface LoaderStats {
  modelId: string | null;
  isLoaded: boolean;
  isMoE: boolean;
  isUnifiedMemory: boolean;
  layersLoaded: number;
  expertsLoaded: number;
  gpuBuffers: number;
}

/**
 * GPU kernel capabilities
 */
export interface KernelCapabilities {
  hasF16: boolean;
  hasSubgroups: boolean;
}

/**
 * Q4K loading configuration.
 */
export interface Q4KConfig {
  /** Use fused Q4K matmul kernels (keeps raw quantized weights) */
  useFusedQ4K: boolean;
  /** Q4K layout: 'row' = fused kernel (fast), 'col' = dequant fallback */
  q4kLayout: 'row' | 'col' | null;
  /** Keep weights as F32 (disable F16 downcasting) */
  keepF32Weights: boolean;
  /** Explicit dense/fused/mixed projection materialization mode */
  q4kMaterializationMode?: 'dense' | 'fused' | 'mixed';
}

/**
 * Model config (flexible structure from manifest)
 */
export interface ModelConfig {
  num_hidden_layers?: number;
  blockCount?: number;
  text_config?: { num_hidden_layers?: number };
  n_layer?: number;
  num_local_experts?: number;
  num_experts?: number;
  architectures?: string[];
  model_type?: string;
  [key: string]: unknown;
}

/**
 * Shard source tracking
 */
export interface ShardSourceInfo {
  source: 'RAM' | 'OPFS' | 'custom' | 'network' | 'indexeddb' | 'memory' | 'storage' | string;
  elapsed: number;
  mode?: 'full' | 'range' | 'stream';
  path?:
    | 'cache'
    | 'custom-loader'
    | 'custom-range'
    | 'custom-stream'
    | 'custom-loader-slice'
    | 'custom-range-fallback'
    | 'backend-full'
    | 'backend-range'
    | 'backend-stream';
  fallback?:
    | 'none'
    | 'custom_range_unavailable'
    | 'custom_range_not_supported'
    | 'custom_stream_not_supported'
    | 'custom_stream_not_supported_resume'
    | 'custom_stream_interrupted'
    | 'custom_stream_interrupted_resume'
    | 'custom_stream_partial_resume'
    | 'custom_range_partial_retry';
}
