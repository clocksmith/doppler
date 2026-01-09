import { DEFAULT_KVCACHE_CONFIG } from './kvcache.schema.js';
import { DEFAULT_MOE_RUNTIME_CONFIG } from './moe.schema.js';

// =============================================================================
// Batching Defaults
// =============================================================================

export const DEFAULT_BATCHING_DEFAULTS = {
  batchSize: 1,  // Compare single-token
  maxTokens: 512,
  stopCheckMode: 'per-token',
};

// =============================================================================
// Compute Defaults
// =============================================================================

export const DEFAULT_COMPUTE_DEFAULTS = {
  activationDtype: 'f32',  // Safe default, F16 is experimental
  largeModelParamThreshold: 4e9,  // 4B parameters
  paramEstimationMultiplier: 12,  // Rough approximation: 12 * hidden^2 * layers
  keepF32Weights: false,  // Skip weight downcast (debug/compat)
};

// =============================================================================
// Large Weight Handling
// =============================================================================

export const DEFAULT_LARGE_WEIGHT_CONFIG = {
  enabled: true,
  safetyRatio: 0.9,
  preferF16: true,
  lmHeadChunkRows: null,
};

// =============================================================================
// Sampling Defaults
// =============================================================================

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

export const DEFAULT_TOKENIZER_DEFAULTS = {
  addBosToken: true,
  addEosToken: false,
};

// =============================================================================
// Complete Inference Defaults Config
// =============================================================================

export const DEFAULT_INFERENCE_DEFAULTS_CONFIG = {
  batching: DEFAULT_BATCHING_DEFAULTS,
  sampling: DEFAULT_SAMPLING_DEFAULTS,
  compute: DEFAULT_COMPUTE_DEFAULTS,
  tokenizer: DEFAULT_TOKENIZER_DEFAULTS,
  largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
  kvcache: DEFAULT_KVCACHE_CONFIG,
  moe: DEFAULT_MOE_RUNTIME_CONFIG,
  prompt: null,
  pipeline: null,
  kernelPath: undefined,
};
