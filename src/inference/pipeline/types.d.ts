/**
 * Shared types for pipeline modules.
 *
 * This module contains all interfaces and types used across pipeline sub-modules.
 * Centralizing types here avoids circular dependencies and provides a single
 * source of truth for the pipeline's type definitions.
 *
 * @module inference/pipeline/types
 */

import type { ParsedModelConfig } from './config.js';
import type { LoRAAdapter } from './lora-types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../gpu/weight-buffer.js';
import type { ProbeConfigSchema } from '../../config/schema/index.js';
import type { ExpertLoader } from './moe-impl.js';
import type { MoERouter } from '../moe-router.js';
import type { DecodeBufferManager } from '../decode-buffers.js';
import type { CommandRecorder } from '../../gpu/kernel-selector.js';
import type { CompiledLayerPipeline } from './layer-plan.js';
import type { WeightBufferConfig, WeightDebugFlags } from './weights.js';
import type { KVCache, SlidingWindowKVCache } from '../kv-cache.js';

// ============================================================================
// Core Context Types
// ============================================================================

export interface KVCacheSnapshot {
  cache: KVCache;
  seqLen: number;
  tokens: number[];
}

/**
 * Layer context contains all state needed for layer processing.
 */
export interface LayerContext {
  /** Model configuration */
  config: ParsedModelConfig;
  /** Layer weights map */
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer | WeightBuffer | CpuWeightBuffer>;
  /** KV cache instance */
  kvCache: KVCache | SlidingWindowKVCache;
  /** Current sequence length */
  currentSeqLen: number;
  /** Whether to use GPU */
  useGPU: boolean;
  /** Debug mode */
  debug: boolean;
  /** Config-driven probes */
  debugProbes?: ProbeConfigSchema[];
  /** Layers to debug (null = none, undefined/empty = layer 0 only for backward compat) */
  debugLayers?: number[] | null;
  /** Optional GPU buffer readback helper for debug checks */
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>;
  /** Optional layer pipeline plan (JSON-configured) */
  pipelinePlan?: CompiledLayerPipeline | null;
  /** RoPE frequency buffers (global for full_attention layers) */
  ropeFreqsCos: GPUBuffer | Float32Array | null;
  ropeFreqsSin: GPUBuffer | Float32Array | null;
  /** Local RoPE frequency buffers for sliding_attention layers (Gemma 3: 10K theta) */
  ropeLocalCos?: GPUBuffer | Float32Array | null;
  ropeLocalSin?: GPUBuffer | Float32Array | null;
  /** Weight buffer config */
  weightConfig: WeightBufferConfig;
  /** Debug flags (mutable) */
  debugFlags?: WeightDebugFlags;
  /** Expert weights map (for MoE) */
  expertWeights?: Map<string, ExpertWeights>;
  /** Expert loader (for MoE) */
  expertLoader?: ExpertLoader | null;
  /** MoE router (for MoE) */
  moeRouter?: MoERouter | null;
  /** Layer router weights (for models with per-layer routers) */
  layerRouterWeights?: Map<number, RouterWeights>;
  /** Command recorder for batched GPU operations (optional) */
  recorder?: CommandRecorder;
  /** Optional LoRA adapter */
  lora?: LoRAAdapter | null;
  /** Pre-allocated decode buffers (for M=1 decode optimization) */
  decodeBuffers?: DecodeBufferManager | null;
  /** Activation dtype for hidden states (default: 'f32', experimental: 'f16') */
  activationDtype?: 'f16' | 'f32';
}

/**
 * Layer processing result.
 */
export interface LayerResult {
  /** Output hidden states (GPUBuffer or Float32Array) */
  output: GPUBuffer | Float32Array;
  /** Whether output is on GPU */
  isGPU: boolean;
}

/**
 * Sandwich norm detection result.
 */
export interface SandwichNormInfo {
  /** Whether sandwich norms are used */
  useSandwichNorm: boolean;
  /** Has pre-feedforward norm */
  hasPreFeedforwardNorm: boolean;
  /** Has post-feedforward norm */
  hasPostFeedforwardNorm: boolean;
  /** Has post-attention norm */
  hasPostAttentionNorm: boolean;
}

/**
 * Pipeline context - all state needed for inference operations.
 *
 * This is passed to all pipeline functions, providing access to:
 * - Model configuration
 * - Weight buffers
 * - KV cache
 * - GPU resources
 * - Runtime state
 */
export interface PipelineContext {
  /** Parsed model configuration */
  config: ParsedModelConfig;

  /** Weight buffers map (name -> GPUBuffer | WeightBuffer | Float32Array) */
  weights: Map<string, GPUBuffer | WeightBuffer | Float32Array>;

  /** KV cache instance */
  kvCache: KVCacheInterface;

  /** Current sequence length (tokens processed so far) */
  currentSeqLen: number;

  /** Whether GPU is available and should be used */
  useGPU: boolean;

  /** Debug mode flag */
  debug: boolean;

  /** RoPE cosine frequencies buffer */
  ropeFreqsCos: GPUBuffer | Float32Array | null;

  /** RoPE sine frequencies buffer */
  ropeFreqsSin: GPUBuffer | Float32Array | null;

  /** Tokenizer instance (for debug logging) */
  tokenizer?: TokenizerInterface;

  /** Statistics tracking */
  stats: PipelineStats;

  /** Batching statistics */
  batchingStats: BatchingStats;

  /** Optional LoRA adapter */
  lora?: LoRAAdapter | null;
}

/** GPU buffer result for KV cache layer */
export interface GPUBuffersResult {
  keysGPU: GPUBuffer;
  valuesGPU: GPUBuffer;
  seqLen: number;
}

/**
 * Minimal KV cache interface for pipeline operations.
 */
export interface KVCacheInterface {
  /** KV cache data type */
  kvDtype?: 'f16' | 'f32';

  /** Get key cache for a layer */
  getKeyCache(layerIdx: number): GPUBuffer | Float32Array | null;

  /** Get value cache for a layer */
  getValueCache(layerIdx: number): GPUBuffer | Float32Array | null;

  /** Update cache for a layer */
  update(
    layerIdx: number,
    position: number,
    keys: GPUBuffer | Float32Array,
    values: GPUBuffer | Float32Array
  ): void;

  /** Clear all cached values */
  clear(): void;

  /** Clone the cache (for speculative decoding) */
  clone?(): KVCacheInterface;

  /** Check if GPU cache is available */
  hasGPUCache?(): boolean;

  /** Update cache from GPU buffers (immediate execution) */
  updateFromGPU?(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void;

  /** Record GPU-based update using command encoder */
  recordUpdateFromGPU?(
    encoder: GPUCommandEncoder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void;

  /** Get GPU buffers for a layer */
  getGPUBuffers?(layerIdx: number): GPUBuffersResult | null;
}

/**
 * Minimal tokenizer interface for pipeline operations.
 */
export interface TokenizerInterface {
  /** Encode text to token IDs */
  encode(text: string): number[];

  /** Decode token IDs to text */
  decode(ids: number[], skipSpecial?: boolean, clean?: boolean): string;

  /** Get special token IDs */
  getSpecialTokens?(): { bos?: number; eos?: number; pad?: number };
}

// ============================================================================
// Generation Types
// ============================================================================

/**
 * Options for text generation.
 */
export interface GenerateOptions {
  /** @category generation Maximum tokens to generate (default: 512) */
  maxTokens?: number;

  /** @category generation Sampling temperature - 0 for greedy (default: 0.7) */
  temperature?: number;

  /** @category generation Top-p (nucleus) sampling threshold (default: 0.9) */
  topP?: number;

  /** @category generation Top-k sampling - 0 to disable (default: 40) */
  topK?: number;

  /** @category generation Repetition penalty multiplier (default: 1.1) */
  repetitionPenalty?: number;

  /** @category generation Stop sequences to end generation */
  stopSequences?: string[];

  /** @category generation Enable speculative decoding */
  useSpeculative?: boolean;

  /** @category hybrid Apply chat template (auto-detected for Gemma) */
  useChatTemplate?: boolean;

  /** Callback for each generated token */
  onToken?: ((tokenId: number, text: string) => void) | null;

  /** Custom decode function for debugging */
  decode?: (tokens: number[]) => string;

  /** Enable debug logging */
  debug?: boolean;

  /** Specific layers to debug */
  debugLayers?: number[];

  /** Abort signal to cancel generation */
  signal?: AbortSignal;

  /** Enable GPU timestamp profiling */
  profile?: boolean;

  /** Log benchmark stats */
  benchmark?: boolean;

  /** Explicitly disable GPU command recording/batching */
  disableCommandBatching?: boolean;

  /** Explicitly disable multi-token GPU decode path */
  disableMultiTokenDecode?: boolean;

  /**
   * @category session Number of tokens to generate per GPU submission batch.
   * @throws DopplerConfigError if set at call-time
   */
  batchSize?: number;

  /** Callback invoked after each batch completes */
  onBatch?: ((tokens: Array<{ id: number; text: string }>) => void) | null;

  /** Stop condition checking mode */
  stopCheckMode?: 'batch' | 'per-token';
}

/**
 * Result of text generation.
 */
export interface GenerationResult {
  /** All token IDs (prompt + generated) */
  tokens: number[];

  /** Generated text (excluding prompt) */
  text: string;

  /** Why generation stopped */
  finishReason: 'stop' | 'length' | 'eos';

  /** Performance statistics */
  stats: {
    prefillTimeMs: number;
    decodeTimeMs: number;
    totalTimeMs: number;
    tokensGenerated: number;
  };
}

// ============================================================================
// Layer Types
// ============================================================================

/**
 * Layer configuration extracted from model config.
 */
export interface LayerConfig {
  layerIdx: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  activation: string;
  useMoE: boolean;
  numExperts?: number;
  topKExperts?: number;
}

/**
 * Weight type that can be a raw GPUBuffer, a typed WeightBuffer, or CPU Float32Array.
 * WeightBuffer provides explicit dtype/layout metadata; GPUBuffer uses WeakMap tracking.
 */
export type LayerWeightBuffer = GPUBuffer | WeightBuffer | Float32Array | CpuWeightBuffer;

/**
 * Weights for a single transformer layer.
 */
export interface LayerWeights {
  // Attention
  inputNorm: GPUBuffer | Float32Array;
  qProj: LayerWeightBuffer;
  kProj: LayerWeightBuffer;
  vProj: LayerWeightBuffer;
  oProj: LayerWeightBuffer;
  /** Fused Q/K/V projection (runtime-generated for 3->1 matmul optimization) */
  qkvProj?: GPUBuffer | WeightBuffer | null;
  /** Sizes for splitting fused QKV output: [qSize, kSize, vSize] in elements */
  qkvSizes?: [number, number, number];
  /** Data type of fused QKV weights (f16 or f32) */
  qkvDtype?: 'f16' | 'f32';

  // FFN (dense layers)
  postAttentionNorm?: GPUBuffer | Float32Array;
  postAttnNorm?: GPUBuffer | Float32Array;  // LLaMA-style pre-FFN norm
  gate?: LayerWeightBuffer;
  up?: LayerWeightBuffer;
  down?: LayerWeightBuffer;
  gateUp?: LayerWeightBuffer;  // Fused gate+up for 2-pass FFN

  // Sandwich norms (Gemma 3)
  preFeedforwardNorm?: GPUBuffer | Float32Array;
  postFeedforwardNorm?: GPUBuffer | Float32Array;

  // MoE
  routerWeight?: GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | Float32Array;
  routerBias?: GPUBuffer | Float32Array | null;
  qNorm?: GPUBuffer | Float32Array;
  kNorm?: GPUBuffer | Float32Array;
  experts?: ExpertWeights[];
}

/**
 * Weights for a single MoE expert.
 */
export interface ExpertWeights {
  gate: LayerWeightBuffer;
  up: LayerWeightBuffer;
  down: LayerWeightBuffer;
}

/**
 * Router weights for MoE layers.
 */
export interface RouterWeights {
  weight: GPUBuffer | Float32Array;
  bias?: GPUBuffer | Float32Array | null;
}

// ============================================================================
// Statistics Types
// ============================================================================

/**
 * Pipeline performance statistics.
 */
export interface PipelineStats {
  /** Total prefill time in milliseconds */
  prefillTimeMs: number;

  /** Total decode time in milliseconds */
  decodeTimeMs: number;

  /** Number of tokens processed in prefill */
  prefillTokens: number;

  /** Number of tokens generated in decode */
  decodeTokens: number;

  /** Memory usage in bytes */
  memoryUsageBytes: number;

  // Fields from pipeline.ts
  tokensGenerated: number;
  totalTimeMs: number;
  gpuTimePrefillMs?: number;
  gpuTimeDecodeMs?: number;
}

/**
 * Batching-specific statistics.
 */
export interface BatchingStats {
  /** Number of batched forward passes */
  batchedForwardCalls: number;

  /** Number of unbatched forward passes */
  unbatchedForwardCalls: number;

  /** Total time in batched mode */
  totalBatchedTimeMs: number;

  /** Total time in unbatched mode */
  totalUnbatchedTimeMs: number;

  /** Number of GPU command submissions */
  gpuSubmissions: number;
}

// ============================================================================
// Helper Types
// ============================================================================

/**
 * Buffer type that can be either GPU or CPU.
 */
export type { MaybeGPUBuffer } from './buffer-types.js';

/**
 * Function to decode token IDs to text (for debug logging).
 */
export type DecodeFunction = (ids: number[]) => string;

/**
 * RoPE (Rotary Position Embedding) options.
 */
export interface RoPEOptions {
  /** Base frequency for RoPE */
  base: number;

  /** Dimension of the embedding */
  dim: number;

  /** Maximum sequence length */
  maxSeqLen: number;

  /** Starting position for RoPE computation */
  startPos?: number;

  /** Scaling configuration */
  scaling?: {
    type: string;
    factor?: number;
    lowFreqFactor?: number;
    highFreqFactor?: number;
    originalMaxSeqLen?: number;
  };
}
