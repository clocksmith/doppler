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

// =============================================================================
// Batching Defaults
// =============================================================================

/** Default batching configuration */
export const DEFAULT_BATCHING_DEFAULTS = {
  batchSize: 1,  // Compare single-token
  maxTokens: 512,
  stopCheckMode: 'per-token',
};

// =============================================================================
// Compute Defaults
// =============================================================================

/** Default compute configuration */
export const DEFAULT_COMPUTE_DEFAULTS = {
  activationDtype: 'f32',  // Safe default, F16 is experimental
  largeModelParamThreshold: 4e9,  // 4B parameters
  paramEstimationMultiplier: 12,  // Rough approximation: 12 * hidden^2 * layers
};

// =============================================================================
// Large Weight Handling
// =============================================================================

/** Default large-weight configuration */
export const DEFAULT_LARGE_WEIGHT_CONFIG = {
  enabled: true,
  safetyRatio: 0.9,
  preferF16: true,
  lmHeadChunkRows: null,
};

// =============================================================================
// Sampling Defaults
// =============================================================================

/** Default sampling configuration */
export const DEFAULT_SAMPLING_DEFAULTS = {
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

/** Default tokenizer configuration */
export const DEFAULT_TOKENIZER_DEFAULTS = {
  addBosToken: true,
  addEosToken: false,
};

// =============================================================================
// Complete Inference Defaults Config
// =============================================================================

/** Default inference configuration */
export const DEFAULT_INFERENCE_DEFAULTS_CONFIG = {
  batching: DEFAULT_BATCHING_DEFAULTS,
  sampling: DEFAULT_SAMPLING_DEFAULTS,
  compute: DEFAULT_COMPUTE_DEFAULTS,
  tokenizer: DEFAULT_TOKENIZER_DEFAULTS,
  largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
  prompt: null,
  pipeline: null,
  kernelPath: undefined,
};
