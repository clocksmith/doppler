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
  batchSize: 8,
  maxTokens: 512,
  stopCheckMode: 'per-token',
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
}

/** Default sampling configuration */
export const DEFAULT_SAMPLING_DEFAULTS: SamplingDefaultsSchema = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1,
  greedyThreshold: 0.01,
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
 * Combines batching, sampling, and tokenizer defaults for the inference pipeline.
 */
export interface InferenceDefaultsConfigSchema {
  batching: BatchingDefaultsSchema;
  sampling: SamplingDefaultsSchema;
  tokenizer: TokenizerDefaultsSchema;
  /** Optional default prompt text for test harnesses */
  prompt?: string | null;
  pipeline?: LayerPipelineSchema | null;
}

/** Default inference configuration */
export const DEFAULT_INFERENCE_DEFAULTS_CONFIG: InferenceDefaultsConfigSchema = {
  batching: DEFAULT_BATCHING_DEFAULTS,
  sampling: DEFAULT_SAMPLING_DEFAULTS,
  tokenizer: DEFAULT_TOKENIZER_DEFAULTS,
  prompt: null,
  pipeline: null,
};
