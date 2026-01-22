/**
 * Inference Defaults Config Schema
 *
 * Default values for inference pipeline: batching, sampling, generation.
 * These defaults are used when no model-specific or user overrides are provided.
 *
 * Note: SamplingDefaultsSchema provides defaults for fields from SamplingSchema
 * (in inference.schema.js), plus greedyThreshold and repetitionPenaltyWindow
 * which are unique to defaults.
 *
 * @module config/schema/inference-defaults
 */

import type {
  ChatTemplateSchema,
  InferenceConfigSchema,
  InferenceKernelOverridesSchema,
  LayerPipelineSchema,
  SamplingSchema,
  TokenizerConfigSchema,
} from './inference.schema.js';
import type { KVCacheConfigSchema } from './kvcache.schema.js';
import type { MoERuntimeConfigSchema } from './moe.schema.js';
import type { KernelPathRef } from './kernel-path.schema.js';
import type { ManifestInferenceSchema } from './manifest.schema.js';
import type { SpeculativeConfigSchema } from './speculative.schema.js';

// Re-export for convenience
export type { ManifestInferenceSchema };

/**
 * Deep partial type for runtime overrides of model-specific inference config.
 * Allows overriding any nested field in ManifestInferenceSchema.
 */
export type ModelInferenceOverrides = {
  [P in keyof ManifestInferenceSchema]?: ManifestInferenceSchema[P] extends object
    ? { [K in keyof ManifestInferenceSchema[P]]?: ManifestInferenceSchema[P][K] }
    : ManifestInferenceSchema[P];
};

/**
 * Default batching configuration for inference.
 *
 * Controls how tokens are batched during generation.
 */
export interface BatchingDefaultsSchema {
  /** Number of sequences to process in parallel (default: 1) */
  batchSize: number;

  /** Maximum tokens to generate per sequence (default: 512) */
  maxTokens: number;

  /** When to check for stop conditions: per-token or per-batch (default: 'batch') */
  stopCheckMode: 'per-token' | 'batch';

  /**
   * Number of GPU decode batches to run before readback.
   * null means read back after every batch (default: 1).
   */
  readbackInterval: number | null;

  /** Ring size for token output buffers (null = disable ring) */
  ringTokens: number | null;

  /** Ring size for per-token stop buffers (null = disable ring) */
  ringStop: number | null;

  /** Ring size for staging readback buffers (null = disable ring) */
  ringStaging: number | null;
}

/** Default batching configuration */
export declare const DEFAULT_BATCHING_DEFAULTS: BatchingDefaultsSchema;

/**
 * Default compute precision configuration.
 *
 * Controls dtype for intermediate activations and compute operations.
 * F16 reduces memory bandwidth by 2x but may have precision implications.
 */
export interface ComputeDefaultsSchema {
  /** Dtype for hidden state activations (default: 'f32', experimental: 'f16') */
  activationDtype: 'f16' | 'f32';

  /** Parameter count threshold for "large model" classification (default: 4e9 = 4B params) */
  largeModelParamThreshold: number;

  /** Multiplier for estimating model params from hidden^2 × layers (default: 12) */
  paramEstimationMultiplier: number;

  /** Keep weights in F32 (skip downcast even when F16 is available) */
  keepF32Weights: boolean;
}

/** Default compute configuration */
export declare const DEFAULT_COMPUTE_DEFAULTS: ComputeDefaultsSchema;

/**
 * Configuration for oversized weights (embeddings, LM head).
 *
 * When weights exceed device binding limits, DOPPLER can keep them on CPU
 * and stream chunks to the GPU for matmul or gather operations.
 */
export interface LargeWeightConfigSchema {
  /** Enable CPU-backed chunking for oversized weights */
  enabled: boolean;
  /** Safety ratio applied to GPU binding limits (0..1). Default: 0.9 */
  safetyRatio: number;
  /** Prefer uploading F16 chunks when supported (reduces chunk size) */
  preferF16: boolean;
  /** Optional override for LM head chunk rows (null = auto) */
  lmHeadChunkRows?: number | null;
}

/** Default large-weight configuration */
export declare const DEFAULT_LARGE_WEIGHT_CONFIG: LargeWeightConfigSchema;

/**
 * Default sampling configuration for token selection.
 *
 * Extends Required<SamplingSchema> with greedyThreshold and repetitionPenaltyWindow
 * for runtime decisions.
 * SamplingSchema (in inference.schema.js) uses optional fields for partial overrides;
 * this schema provides concrete defaults for all sampling parameters.
 */
export interface SamplingDefaultsSchema extends Required<SamplingSchema> {
  /** Temperature below this uses greedy decoding (default: 0.01) */
  greedyThreshold: number;

  /** Number of recent tokens to consider for repetition penalty (default: 100) */
  repetitionPenaltyWindow: number;
}

/** Default sampling configuration */
export declare const DEFAULT_SAMPLING_DEFAULTS: SamplingDefaultsSchema;

/**
 * Default tokenizer configuration.
 *
 * Provides defaults for common tokenizer behavior. Actual token strings
 * come from the model's tokenizer config; these control runtime behavior.
 */
export interface TokenizerDefaultsSchema {
  /** Add BOS token to input (default: true for most models) */
  addBosToken: boolean;

  /** Add EOS token to output (default: false, model decides) */
  addEosToken: boolean;
}

/** Default tokenizer configuration */
export declare const DEFAULT_TOKENIZER_DEFAULTS: TokenizerDefaultsSchema;

/**
 * Complete inference defaults configuration schema.
 *
 * Combines batching, sampling, compute, and tokenizer defaults for the inference pipeline.
 */
export interface InferenceDefaultsConfigSchema {
  batching: BatchingDefaultsSchema;
  sampling: SamplingDefaultsSchema;
  compute: ComputeDefaultsSchema;
  tokenizer: TokenizerDefaultsSchema;
  /** Handling for oversized embeddings/LM head */
  largeWeights: LargeWeightConfigSchema;
  /** KV cache configuration */
  kvcache: KVCacheConfigSchema;
  /** MoE routing and cache configuration */
  moe: MoERuntimeConfigSchema;
  /** Speculative decoding configuration */
  speculative: SpeculativeConfigSchema;
  /** Optional default prompt text for test harnesses */
  prompt?: string | null;
  /** Log token ids in test harness output */
  debugTokens?: boolean;
  pipeline?: LayerPipelineSchema | null;
  /**
   * Kernel path for explicit kernel dispatch ordering.
   * Specifies exactly which kernels run, in what order, with what configs.
   * Can be a preset ID (e.g., 'gemma2-q4k-fused-f32a') or inline KernelPathSchema.
   */
  kernelPath?: KernelPathRef;
  /** Optional kernel overrides for targeted variants. */
  kernelOverrides?: InferenceKernelOverridesSchema | null;
  /**
   * Chat template override for runtime config.
   * When set, overrides the model preset's chatTemplate.enabled setting.
   */
  chatTemplate?: ChatTemplateSchema;
  /**
   * Model-specific inference overrides.
   * Allows runtime override of any manifest inference field (attention, normalization, rope, etc.).
   * These are merged with manifest.inference via mergeConfig().
   */
  modelOverrides?: ModelInferenceOverrides;
}

/** Default inference configuration */
export declare const DEFAULT_INFERENCE_DEFAULTS_CONFIG: InferenceDefaultsConfigSchema;

/** Default inference configuration for model presets */
export declare const DEFAULT_PRESET_INFERENCE_CONFIG: InferenceConfigSchema;
