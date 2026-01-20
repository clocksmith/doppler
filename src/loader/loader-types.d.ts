/**
 * Loader Types
 *
 * Type definitions for the DopplerLoader.
 *
 * @module loader/loader-types
 */

import type { WeightBuffer } from '../gpu/weight-buffer.js';
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
  qNorm: GPUBuffer | Float32Array | null;
  kNorm: GPUBuffer | Float32Array | null;
  postAttentionNorm: GPUBuffer | Float32Array | null;
  preFeedforwardNorm: GPUBuffer | Float32Array | null;
  postFeedforwardNorm: GPUBuffer | Float32Array | null;
  postNorm: GPUBuffer | Float32Array | null;
  postAttnNorm: GPUBuffer | Float32Array | null;
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
}

/**
 * Custom shard loader function
 */
export type CustomShardLoader = (shardIndex: number) => Promise<Uint8Array>;

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
  source: 'RAM' | 'OPFS' | 'custom' | 'network';
  elapsed: number;
}
