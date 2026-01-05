/**
 * Inference Defaults Config Schema
 *
 * Default values for inference pipeline: batching, sampling, generation.
 * These defaults are used when no model-specific or user overrides are provided.
 *
 * Note: SamplingDefaultsSchema provides defaults for fields from SamplingSchema
 * (in inference.schema.ts), plus the greedyThreshold which is unique to defaults.
 *
 * @module config/schema/inference-defaults
 */

import type { LayerPipelineSchema, SamplingSchema, TokenizerConfigSchema } from './inference.schema.js';
import type { KernelPlanSchema } from './kernel-plan.schema.js';

// =============================================================================
// Batching Defaults
// =============================================================================

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

  /** When to check for stop conditions: per-token or per-batch (default: 'per-token') */
  stopCheckMode: 'per-token' | 'batch';
}

/** Default batching configuration */
export const DEFAULT_BATCHING_DEFAULTS: BatchingDefaultsSchema = {
  batchSize: 1,  // Compare single-token
  maxTokens: 512,
  stopCheckMode: 'per-token',
};

// =============================================================================
// Compute Defaults
// =============================================================================

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
}

/** Default compute configuration */
export const DEFAULT_COMPUTE_DEFAULTS: ComputeDefaultsSchema = {
  activationDtype: 'f32',  // Safe default, F16 is experimental
  largeModelParamThreshold: 4e9,  // 4B parameters
  paramEstimationMultiplier: 12,  // Rough approximation: 12 * hidden^2 * layers
};

// =============================================================================
// Large Weight Handling
// =============================================================================

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
export const DEFAULT_LARGE_WEIGHT_CONFIG: LargeWeightConfigSchema = {
  enabled: true,
  safetyRatio: 0.9,
  preferF16: true,
  lmHeadChunkRows: null,
};

// =============================================================================
// Sampling Defaults
// =============================================================================

/**
 * Default sampling configuration for token selection.
 *
 * Extends Required<SamplingSchema> with greedyThreshold for runtime decisions.
 * SamplingSchema (in inference.schema.ts) uses optional fields for partial overrides;
 * this schema provides concrete defaults for all sampling parameters.
 */
export interface SamplingDefaultsSchema extends Required<Omit<SamplingSchema, 'maxTokens'>> {
  /** Temperature below this uses greedy decoding (default: 0.01) */
  greedyThreshold: number;

  /** Number of recent tokens to consider for repetition penalty (default: 100) */
  repetitionPenaltyWindow: number;
}

/** Default sampling configuration */
export const DEFAULT_SAMPLING_DEFAULTS: SamplingDefaultsSchema = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1,
  greedyThreshold: 0.01,
  repetitionPenaltyWindow: 100,
};

// =============================================================================
// Tokenizer Defaults
// =============================================================================

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
export const DEFAULT_TOKENIZER_DEFAULTS: TokenizerDefaultsSchema = {
  addBosToken: true,
  addEosToken: false,
};

// =============================================================================
// Complete Inference Defaults Config
// =============================================================================

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
  /** Optional default prompt text for test harnesses */
  prompt?: string | null;
  pipeline?: LayerPipelineSchema | null;
  /** Kernel pipeline plan overrides */
  kernelPlan?: KernelPlanSchema | null;
}

/** Default inference configuration */
export const DEFAULT_INFERENCE_DEFAULTS_CONFIG: InferenceDefaultsConfigSchema = {
  batching: DEFAULT_BATCHING_DEFAULTS,
  sampling: DEFAULT_SAMPLING_DEFAULTS,
  compute: DEFAULT_COMPUTE_DEFAULTS,
  tokenizer: DEFAULT_TOKENIZER_DEFAULTS,
  largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
  prompt: null,
  pipeline: null,
  kernelPlan: null,
};
