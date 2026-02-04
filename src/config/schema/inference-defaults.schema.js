import { DEFAULT_KVCACHE_CONFIG } from './kvcache.schema.js';
import { DEFAULT_MOE_RUNTIME_CONFIG } from './moe.schema.js';
import { DEFAULT_SPECULATIVE_CONFIG } from './speculative.schema.js';
import { DEFAULT_RMS_NORM_EPS } from './manifest.schema.js';
import { DEFAULT_DIFFUSION_CONFIG } from './diffusion.schema.js';
import { DEFAULT_ENERGY_CONFIG } from './energy.schema.js';

// =============================================================================
// Generation Defaults (generator.js runtime options)
// =============================================================================

export const DEFAULT_GENERATION_CONFIG = {
  useSpeculative: false,
  profile: false,
  benchmark: false,
  disableCommandBatching: false,
  disableMultiTokenDecode: false,
};

// =============================================================================
// Batching Defaults
// =============================================================================

export const DEFAULT_BATCHING_DEFAULTS = {
  batchSize: 1,  // Compare single-token
  maxTokens: 512,
  stopCheckMode: 'batch',
  readbackInterval: 1,
  ringTokens: 1,
  ringStop: 1,
  ringStaging: 1,
};

// =============================================================================
// Compute Defaults
// =============================================================================

export const DEFAULT_COMPUTE_DEFAULTS = {
  activationDtype: 'f16',  // Default to F16 for web inference; fallback to F32 when unsupported
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
  addSpacePrefix: null,  // null = auto-detect from tokenizer.json
};

// =============================================================================
// Chat Template Defaults
// =============================================================================

export const DEFAULT_CHAT_TEMPLATE_CONFIG = {
  enabled: false,
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
  speculative: DEFAULT_SPECULATIVE_CONFIG,
  generation: DEFAULT_GENERATION_CONFIG,
  chatTemplate: DEFAULT_CHAT_TEMPLATE_CONFIG,
  diffusion: DEFAULT_DIFFUSION_CONFIG,
  energy: DEFAULT_ENERGY_CONFIG,
  pipeline: null,
  kernelPath: undefined,
  kernelOverrides: null,
};

// =============================================================================
// Preset Inference Defaults
// =============================================================================

export const DEFAULT_PRESET_INFERENCE_CONFIG = {
  attention: {
    slidingWindow: null,
    attnLogitSoftcapping: null,
    queryKeyNorm: false,
    ropeScalingType: null,
    ropeScalingFactor: 1.0,
  },
  normalization: {
    rmsNormWeightOffset: false,
    rmsNormEps: DEFAULT_RMS_NORM_EPS,
    postAttentionNorm: false,
    preFeedforwardNorm: false,
    postFeedforwardNorm: false,
  },
  ffn: {
    activation: 'silu',
    gatedActivation: true,
  },
  output: {
    finalLogitSoftcapping: null,
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
  },
  layerPattern: {
    type: 'all_attention',
  },
  rope: {
    ropeTheta: 10000,
    ropeLocalTheta: null,
    ropeScalingType: null,
    ropeScalingFactor: 1.0,
    yarnBetaFast: 32,
    yarnBetaSlow: 1,
    yarnOriginalMaxPos: 4096,
  },
  pipeline: null,
  chatTemplate: {
    type: null,
  },
  kernelPath: undefined,
};
