/**
 * Loader Types
 *
 * Type definitions for the DopplerLoader.
 *
 * @module loader/loader-types
 */

import type { CpuWeightBuffer, WeightBuffer } from '../gpu/weight-buffer.js';
import type { TensorRole } from '../config/schema/index.js';
import type { TensorSourceTransform } from '../formats/rdrr/index.js';
import type { FunctionalDescriptorManifest } from '../formats/rdrr/functional-descriptor.js';

/**
 * Tensor location in loaded model
 */
export interface TensorLocation {
  shardIndex?: number;
  offset?: number;
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
  /** Optional direct-source transform applied before dtype-specific loading */
  sourceTransform?: TensorSourceTransform;
  /** Parsed manifoldgguf descriptor for FUNCTIONAL_DESCRIPTOR tensors */
  descriptorManifest?: FunctionalDescriptorManifest;
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
  preFeedforwardNorm2?: GPUBuffer | Float32Array | null;
  postFeedforwardNorm: GPUBuffer | Float32Array | null;
  postFeedforwardNorm1?: GPUBuffer | Float32Array | null;
  postFeedforwardNorm2?: GPUBuffer | Float32Array | null;
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
  routerScale?: GPUBuffer | Float32Array | null;
  routerPerExpertScale?: GPUBuffer | Float32Array | null;
  attentionSinks?: GPUBuffer | Float32Array | null;
  perLayerInputGate?: GPUBuffer | WeightBuffer | Float32Array | null;
  perLayerProjection?: GPUBuffer | WeightBuffer | Float32Array | null;
  postPerLayerInputNorm?: GPUBuffer | Float32Array | null;
  layerScalar?: GPUBuffer | Float32Array | null;
}

export interface DiffusionGemmaSelfConditioningWeights {
  preNorm: GPUBuffer | Float32Array;
  postNorm: GPUBuffer | Float32Array | null;
  gateProj: GPUBuffer | WeightBuffer | Float32Array;
  upProj: GPUBuffer | WeightBuffer | Float32Array;
  downProj: GPUBuffer | WeightBuffer | Float32Array;
}

export interface PerLayerInputWeights {
  embedTokensPerLayer: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  embedTokensPerLayerSplit?: (GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null)[] | null;
  perLayerModelProjection: GPUBuffer | WeightBuffer | Float32Array | null;
  perLayerProjectionNorm: GPUBuffer | Float32Array | null;
  perLayerModelProjectionInputActivationStaticScale?: number | null;
  perLayerModelProjectionOutputActivationStaticScale?: number | null;
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

export type LoaderLoadTimingPhase =
  | 'preflight'
  | 'tensorLocations'
  | 'embeddings'
  | 'layers'
  | 'finalWeights'
  | 'cleanup';

export interface LoaderLayerLoadTiming {
  count: number | null;
  totalMs: number | null;
  meanMs: number | null;
  maxMs: number | null;
  maxLayer: number | null;
}

export interface LoaderLoadTiming {
  schemaVersion: 1;
  source: 'doppler-loader';
  modelId: string | null;
  status: 'running' | 'complete' | 'failed';
  customShardLoader: boolean;
  byteAccountingMode: 'full-shard-progress' | 'custom-loader-progress-unavailable';
  totalBytes: number | null;
  totalShards: number | null;
  bytesLoaded: number;
  shardsLoaded: number;
  bytesPerSecond: number | null;
  phasesMs: Record<LoaderLoadTimingPhase, number | null>;
  layers: LoaderLayerLoadTiming;
  totalMs: number | null;
  failedPhase: LoaderLoadTimingPhase | null;
  error: string | null;
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
  loadAuxiliaryFile?: CustomAuxiliaryFileLoader | null;
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

export type CustomAuxiliaryFileLoader = (
  path: string
) => Promise<ArrayBuffer | Uint8Array | null | undefined>;

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
  loadTiming?: LoaderLoadTiming | null;
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
  /** Tensor roles that must keep raw Q4K materialization because the execution graph pins a fused Q4K kernel */
  q4kFusedRoles?: string[];
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
