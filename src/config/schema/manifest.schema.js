import { MB } from './units.schema.js';

// =============================================================================
// Hash & Versioning
// =============================================================================

export const RDRR_VERSION = 1;

export const SHARD_SIZE = 64 * MB;

export const TENSORS_FILENAME = 'tensors.json';

// =============================================================================
// Parser Constants
// =============================================================================

// Maximum header size for model format parsing.
// GGUF/SafeTensors headers typically fit in first 100MB even for huge models.
export const MAX_HEADER_SIZE = 100 * MB;

// Smaller header read for streaming/browser imports (10MB covers typical headers)
export const HEADER_READ_SIZE = 10 * MB;

// =============================================================================
// Epsilon Constants
// =============================================================================

// Default RMS normalization epsilon - used across all model types
export const DEFAULT_RMS_NORM_EPS = 1e-5;

// Higher precision epsilon for numerical stability in some operations
export const DEFAULT_HIGH_PRECISION_EPS = 1e-6;

// =============================================================================
// Inference Schema (Model-Specific Inference Parameters)
// =============================================================================

export const DEFAULT_MANIFEST_INFERENCE = {
  presetId: null,
  attention: {
    queryPreAttnScalar: 8,  // sqrt(64) for standard 64-dim heads
    attnLogitSoftcapping: null,  // No softcapping (null = disabled)
    slidingWindow: null,  // Full attention (null = no sliding window)
    queryKeyNorm: false,
    causal: true,  // Causal mask enabled by default (decoder-style attention)
    attentionBias: false,
  },
  normalization: {
    rmsNormEps: DEFAULT_RMS_NORM_EPS,
    rmsNormWeightOffset: false,
    postAttentionNorm: false,
    preFeedforwardNorm: false,
    postFeedforwardNorm: false,
  },
  ffn: {
    activation: 'silu',
    gatedActivation: true,
    swigluLimit: null,
  },
  rope: {
    ropeTheta: 10000,
    ropeLocalTheta: null,  // Same as ropeTheta (null = use ropeTheta)
    ropeScalingType: null,  // No scaling (null = disabled)
    ropeScalingFactor: 1.0,
    // YARN parameters - only relevant when ropeScalingType='yarn'
    yarnBetaFast: 32,
    yarnBetaSlow: 1,
    yarnOriginalMaxPos: 4096,
  },
  output: {
    finalLogitSoftcapping: null,  // No softcapping (null = disabled)
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
  },
  layerPattern: {
    type: 'uniform',  // All layers same type
    globalPattern: null,  // No alternating pattern (null = not applicable)
    period: null,  // No periodic pattern (null = not applicable)
    offset: null,  // For every_n: first global layer index modulo period
  },
  chatTemplate: {
    type: null,  // No chat template (null = disabled)
    enabled: false,
  },
  pipeline: null,
  defaultKernelPath: null,  // Use default kernel selection (null = no explicit path)
};

// =============================================================================
// Validation Helpers
// =============================================================================

export function isV1Manifest(manifest) {
  return manifest.version === 1 && !!manifest.groups;
}

export function hasMoEConfig(manifest) {
  return manifest.moeConfig != null && manifest.moeConfig.numExperts > 1;
}

export function validateManifestInference(
  manifest
) {
  if (!manifest.inference) {
    throw new Error(
      `Manifest for "${manifest.modelId}" is missing required 'inference' field. ` +
      `This model was converted with an older version of DOPPLER. ` +
      `Please re-convert the model using the latest converter.`
    );
  }
}

export function hasInferenceConfig(
  manifest
) {
  return manifest.inference != null;
}
