/**
 * Shared types for pipeline modules.
 *
 * This module contains all interfaces and types used across pipeline sub-modules.
 * Centralizing types here avoids circular dependencies and provides a single
 * source of truth for the pipeline's type definitions.
 *
 * @module inference/pipelines/text/types
 */

import type { ParsedModelConfig } from './config.js';
import type { LoRAAdapter } from './lora-types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { ProbeConfigSchema, KernelPathSchema } from '../../../config/schema/index.js';
import type { ExpertLoader } from './moe-impl.js';
import type { MoERouter } from '../../moe-router.js';
import type { DecodeBufferManager } from '../../decode-buffers.js';
import type { CommandRecorder } from '../../../gpu/kernel-selector.js';
import type { CompiledLayerPipeline } from './layer-plan.js';
import type { WeightBufferConfig, WeightDebugFlags } from './weights.js';
import type {
  ExecutionV1PerLayerInputsSessionSchema,
  ExecutionV1PoliciesSchema,
} from '../../../config/schema/execution-v1.schema.js';
import type {
  KVCache,
  SlidingWindowKVCache,
  TieredKVCache,
  BasisDecomposedPagedCache,
  MixedGeometryKVCache,
} from '../../kv-cache.js';
import type { DecodeRingStats } from '../../decode-ring.js';
import type { LinearAttentionRuntime } from './linear-attention.js';
import type { PerLayerInputWeights } from '../../../loader/loader-types.js';

// ============================================================================
// Core Context Types
// ============================================================================

export interface KVCacheSnapshot {
  cache: KVCache | SlidingWindowKVCache | TieredKVCache | BasisDecomposedPagedCache | MixedGeometryKVCache;
  seqLen: number;
  tokens: number[];
  linearAttention?: LinearAttentionRuntime | null;
}

export interface AdvanceEmbeddingResult {
  embedding: Float32Array;
  embeddingMode: 'last' | 'mean';
  seqLen: number;
}

/**
 * Layer context contains all state needed for layer processing.
 */
export interface LayerContext {
  /** Model configuration */
  config: ParsedModelConfig;
  /** Layer weights map */
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer | WeightBuffer | CpuWeightBuffer | PerLayerInputWeights | null>;
  /** KV cache instance */
  kvCache: KVCache | SlidingWindowKVCache | TieredKVCache | BasisDecomposedPagedCache | MixedGeometryKVCache;
  /** Recurrent runtime state for linear_attention layers */
  linearAttentionRuntime?: LinearAttentionRuntime | null;
  /** Current sequence length */
  currentSeqLen: number;
  /** Token IDs for the current micro-batch (required by BDPA ingestion). */
  currentTokenIds?: number[] | null;
  /** Absolute-position multimodal span that should remain bidirectional during causal prefill. */
  multimodalBidirectionalSpan?: {
    start: number;
    length: number;
  } | null;
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
  /** Gemma 4 per-layer input buffer for the active decoder layer. */
  perLayerInputBuffer?: GPUBuffer | null;
  /** Resolved session policy for Gemma 4 per-layer input materialization/caching. */
  perLayerInputsSession?: ExecutionV1PerLayerInputsSessionSchema | null;
  /** Pipeline stats surface for cache hit/miss accounting. */
  stats?: PipelineStats;
  /** Optional layer pipeline plan (JSON-configured) */
  pipelinePlan?: CompiledLayerPipeline | null;
  /** RoPE frequency buffers (global for full_attention layers) */
  ropeFreqsCos: GPUBuffer | Float32Array | null;
  ropeFreqsSin: GPUBuffer | Float32Array | null;
  /** Local RoPE frequency buffers for sliding_attention layers (Gemma 3: 10K theta) */
  ropeLocalCos?: GPUBuffer | Float32Array | null;
  ropeLocalSin?: GPUBuffer | Float32Array | null;
  /** Per-pass shared K/V state for architectures that reuse K/V across later layers. */
  sharedAttentionState?: Map<number, {
    kTensor: unknown;
    vTensor: unknown;
    headDim: number;
    numKVHeads: number;
  }>;
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
  /** Runtime compute config snapshot for layer execution. */
  runtimeComputeConfig?: {
    activationDtype?: 'f16' | 'f32';
    deferredRoundingWindowTokens?: number;
    rangeAwareSelectiveWidening?: {
      enabled?: boolean;
      includeNonFinite?: boolean;
      onTrigger?: 'error' | 'fallback-plan';
      absThreshold?: number;
    };
  };
  /** Activation dtype for hidden states (default: 'f32', experimental: 'f16') */
  activationDtype?: 'f16' | 'f32';
  /** Execution-v1 fused-FFN step precision propagated from the active layer-pipeline step. */
  ffnStepPrecision?: {
    inputDtype?: 'f16' | 'f32' | null;
    outputDtype?: 'f16' | 'f32' | null;
  } | null;
  /** Explicit kernel-path context for kernel variant selection. */
  kernelPath?: KernelPathSchema | null;
  /** Execution-v1 fail-fast policies when the runtime is driven by an execution graph. */
  executionPolicies?: ExecutionV1PoliciesSchema | null;
  /** Shared finiteness status buffer for always-on guard checks */
  finitenessBuffer?: GPUBuffer | null;
  /** Enable/disable guard dispatch for this layer context */
  finitenessGuardEnabled?: boolean;
  /** Absolute-value threshold used by finiteness guard */
  finitenessAbsThreshold?: number;
  /** Decode step index used for first-hit metadata */
  step?: number;
  /** Semantic phase for operator diagnostics */
  phase?: 'prefill' | 'decode' | null;
  /** Operator diagnostics state */
  operatorDiagnostics?: {
    enabled?: boolean;
    emitter?: {
      emitRecord(stageName: string, options?: Record<string, unknown>): unknown;
    };
    captureConfig?: Record<string, unknown>;
  } | null;
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

/** GPU buffer result for KV cache layer */
export interface GPUBuffersResult {
  keysGPU?: GPUBuffer;
  valuesGPU?: GPUBuffer;
  seqLen: number;
  layout?: 'contiguous' | 'ring' | 'paged' | 'tiered' | 'bdpa';
  pageTableGPU?: GPUBuffer;
  pageSize?: number;
  hotKeysGPU?: GPUBuffer;
  hotValuesGPU?: GPUBuffer;
  hotSeqLen?: number;
  hotStart?: number;
  hotWindow?: number;
  coldKeysGPU?: GPUBuffer;
  coldValuesGPU?: GPUBuffer;
  coldScalesKGPU?: GPUBuffer;
  coldScalesVGPU?: GPUBuffer;
  coldSeqLen?: number;
  coldPageTableGPU?: GPUBuffer;
  coldPageSize?: number;
  coldPackedStride?: number;
  coldQuantMode?: 'none' | 'int8' | 'int4';
  basisGPU?: { k: GPUBuffer; v: GPUBuffer };
  pagedGPU?: { k: GPUBuffer; v: GPUBuffer };
  indexGPU?: GPUBuffer;
  numBasisVectors?: number;
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
    numTokens: number,
    tokenIds?: number[] | null
  ): void | Promise<void>;

  /** Record GPU-based update using command encoder */
  recordUpdateFromGPU?(
    recorder: CommandRecorder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number,
    tokenIds?: number[] | null
  ): void | Promise<void>;

  /** Get GPU buffers for a layer */
  getGPUBuffers?(layerIdx: number): GPUBuffersResult | null;
}

// ============================================================================
// Generation Types
// ============================================================================

/**
 * Options for text generation.
 */
export interface PrefixEmbeddingOverride {
  /** Number of token embeddings to replace. */
  prefixLength: number;

  /** Token offset where the replacement span starts. Defaults to 0. */
  offset?: number;

  /** Replacement embeddings as packed f32 activations. */
  embeddings: Float32Array | GPUBuffer;
}

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

  /** @internal Callback with finalized per-step logits before sampling returns the token. */
  onLogits?: ((logits: Float32Array, context: {
    tokenId: number;
    inputTokenCount: number | null;
  }) => void) | null;

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

  /**
   * @category prefill When using prefill helpers that return an embedding, controls pooling.
   * - 'last': last-token hidden state (default)
   * - 'mean': mean-pooled token hidden states (slower; requires extra readback)
   */
  embeddingMode?: 'last' | 'mean';

  /** @category prefill Explicit token IDs to use instead of encoding the prompt string. */
  inputIds?: readonly number[] | Int32Array | Uint32Array;

  /** @category prefill Replace a contiguous input span with explicit embedding activations. */
  embeddingOverrides?: PrefixEmbeddingOverride | null;

  /** @internal Preserve a multimodal span that must remain bidirectional within causal prefill attention. */
  __internalMultimodalBidirectionalSpan?: {
    offset: number;
    length: number;
  } | null;
}

/**
 * Result of a logits-only decode step.
 */
export interface LogitsStepResult {
  /** Finalized logits for the next token */
  logits: Float32Array;

  /** Vocabulary size for finalized logits */
  vocabSize: number;

  /** Raw vocab size from the LM head matmul */
  rawVocabSize: number;

  /** Optional GPU buffer containing raw logits */
  logitsBuffer?: GPUBuffer | null;

  /** Dtype of logitsBuffer when present */
  logitsDtype?: string | null;
}

/**
 * Result of prefill with logits.
 */
export interface PrefillResult extends KVCacheSnapshot {
  /** Finalized logits for the next token after prefill */
  logits: Float32Array;
}

/**
 * Result of prefill that returns a compact intent embedding (no logits).
 *
 * This is the prefill-first "read a lot, output a little" fast path used for:
 * - intent embedding
 * - retrieval scoring against catalog descriptors
 *
 * It avoids LM-head logits computation, so it is cheaper than PrefillResult.
 */
export interface PrefillEmbeddingResult extends KVCacheSnapshot {
  /** Intent embedding vector (Float32), typically last-token hidden state */
  embedding: Float32Array;

  /** Pooling mode used to construct embedding */
  embeddingMode: 'last' | 'mean';
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
  convInProj?: LayerWeightBuffer;
  convKernel?: LayerWeightBuffer;
  convOutProj?: LayerWeightBuffer;
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
  routerWeight?: GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer | Float32Array;
  routerBias?: GPUBuffer | Float32Array | null;
  qNorm?: GPUBuffer | Float32Array;
  kNorm?: GPUBuffer | Float32Array;
  experts?: ExpertWeights[];
}

/**
 * Weights for a single MoE expert.
 */
export interface ExpertWeights {
  expertFormat?: 'mixtral' | 'gpt-oss';
  gate?: LayerWeightBuffer;
  up?: LayerWeightBuffer;
  down?: LayerWeightBuffer;
  numExperts?: number;
  gateUpBlocks?: GPUBuffer;
  gateUpScales?: GPUBuffer;
  gateUpBias?: GPUBuffer;
  downBlocks?: GPUBuffer;
  downScales?: GPUBuffer;
  downBias?: GPUBuffer;
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

  /** Time to first token in milliseconds */
  ttftMs: number;

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
  decodeRecordMs?: number;
  decodeSubmitWaitMs?: number;
  decodeReadbackWaitMs?: number;
  decodeMode?: 'single_token' | 'batched_gpu' | 'batched_gpu_stepwise_ple' | null;
  batchGuardReason?: string | null;
  singleTokenSubmitWaitMs?: number;
  singleTokenReadbackWaitMs?: number;
  singleTokenOrchestrationMs?: number;
  batching?: BatchingStats | null;
  plePreparedTokenCacheHits?: number;
  plePreparedTokenCacheMisses?: number;
  plePreparedTokenCacheEntries?: number;
  plePreparedTokenCacheBytes?: number;
  pleHotVocabularyHits?: number;
  pleHotVocabularyMisses?: number;
  decodeRing?: DecodeRingStats | null;
  executionPlan?: {
    primary: {
      id: string;
      kernelPathId: string | null;
      kernelPathSource: string;
      activationDtype: string;
      readbackInterval: number | null;
      batchSize: number;
      stopCheckMode: string;
      ringTokens: number | null;
      ringStop: number | null;
      ringStaging: number | null;
    } | null;
    fallback: {
      id: string;
      kernelPathId: string | null;
      kernelPathSource: string;
      activationDtype: string;
      readbackInterval: number | null;
      batchSize: number;
      stopCheckMode: string;
      ringTokens: number | null;
      ringStop: number | null;
      ringStaging: number | null;
    } | null;
    activePlanIdAtStart: string | null;
    finalActivePlanId: string | null;
    transitions: Array<{
      kind: string;
      reason: string | null;
      decodeStep: number;
      seqLen: number;
      fromPlanId: string | null;
      toPlanId: string | null;
      fromKernelPathId: string | null;
      toKernelPathId: string | null;
    }>;
  } | null;
  kernelPathId?: string | null;
  kernelPathSource?: string | null;
  prefillProfileSteps?: Array<{
    label?: string;
    timings: Record<string, number>;
    totalMs?: number;
  }>;
  decodeProfileSteps?: Array<{
    step?: number;
    stepStart?: number;
    stepCount?: number;
    batch?: boolean;
    timings: Record<string, number>;
    totalMs?: number;
  }>;
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
