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

// ============================================================================
// Core Context Types
// ============================================================================

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

  /** Weight buffers map (name -> GPUBuffer | Float32Array) */
  weights: Map<string, GPUBuffer | Float32Array>;

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

  /** Override for attention kernel selection */
  attentionKernelOverride: string | null;

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
  /** Maximum tokens to generate (default: 512) */
  maxTokens?: number;

  /** Sampling temperature - 0 for greedy (default: 0.7) */
  temperature?: number;

  /** Top-p (nucleus) sampling threshold (default: 0.9) */
  topP?: number;

  /** Top-k sampling - 0 to disable (default: 40) */
  topK?: number;

  /** Repetition penalty multiplier (default: 1.1) */
  repetitionPenalty?: number;

  /** Stop sequences to end generation */
  stopSequences?: string[];

  /** Enable speculative decoding */
  useSpeculative?: boolean;

  /** Apply chat template (auto-detected for Gemma) */
  useChatTemplate?: boolean;

  /** Callback for each generated token */
  onToken?: (tokenId: number, text: string) => void;
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
 * Weights for a single transformer layer.
 */
export interface LayerWeights {
  // Attention
  inputNorm: GPUBuffer | Float32Array;
  qProj: GPUBuffer | Float32Array;
  kProj: GPUBuffer | Float32Array;
  vProj: GPUBuffer | Float32Array;
  oProj: GPUBuffer | Float32Array;
  /** Fused Q/K/V projection (runtime-generated for 3→1 matmul optimization) */
  qkvProj?: GPUBuffer | null;
  /** Sizes for splitting fused QKV output: [qSize, kSize, vSize] in elements */
  qkvSizes?: [number, number, number];
  /** Data type of fused QKV weights (f16 or f32) */
  qkvDtype?: 'f16' | 'f32';

  // FFN (dense layers)
  postAttentionNorm?: GPUBuffer | Float32Array;
  postAttnNorm?: GPUBuffer | Float32Array;  // LLaMA-style pre-FFN norm
  gate?: GPUBuffer | Float32Array;
  up?: GPUBuffer | Float32Array;
  down?: GPUBuffer | Float32Array;
  gateUp?: GPUBuffer | Float32Array;  // Fused gate+up for 2-pass FFN

  // Sandwich norms (Gemma 3)
  preFeedforwardNorm?: GPUBuffer | Float32Array;
  postFeedforwardNorm?: GPUBuffer | Float32Array;

  // MoE
  routerWeight?: GPUBuffer | Float32Array;
  routerBias?: GPUBuffer | Float32Array | null;
  qNorm?: GPUBuffer | Float32Array;
  kNorm?: GPUBuffer | Float32Array;
  experts?: ExpertWeights[];
}

/**
 * Weights for a single MoE expert.
 */
export interface ExpertWeights {
  gate: GPUBuffer | Float32Array;
  up: GPUBuffer | Float32Array;
  down: GPUBuffer | Float32Array;
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
