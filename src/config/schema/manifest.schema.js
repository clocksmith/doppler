import { MB } from './units.schema.js';
import { validateRequiredInferenceFields } from '../../inference/pipelines/text/config.js';

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

// Smaller header read for streaming/browser imports. Originally 10 MB covered
// typical headers, but Gemma 4 E2B GGUF metadata (262144-token vocab stored as
// inline strings) overflows 10 MB. Bumped to 50 MB to cover large-vocab models
// while keeping Node-side reads bounded.
export const HEADER_READ_SIZE = 50 * MB;

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
  schema: null,
  attention: {
    queryPreAttnScalar: 64, // headDim for standard 64-dim heads; attnScale = 1/sqrt(scalar)
    attnLogitSoftcapping: null,  // No softcapping (null = disabled)
    slidingWindow: null,  // Full attention (null = no sliding window)
    queryKeyNorm: false,
    valueNorm: false,
    causal: true,  // Causal mask enabled by default (decoder-style attention)
    attentionBias: false,
    attentionOutputGate: false,
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
    useDoubleWideMlp: false,
    swigluLimit: null,
  },
  rope: {
    ropeTheta: 10000,
    ropeLocalTheta: null,  // Same as ropeTheta (null = use ropeTheta)
    ropeInterleaved: false,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeLocalPartialRotaryFactor: null,
    ropeFrequencyBaseDim: null,
    ropeLocalFrequencyBaseDim: null,
    ropeScalingType: null,  // No scaling (null = disabled)
    ropeScalingFactor: 1.0,
    ropeLocalScalingType: null,  // Local scaling policy (null = no scaling)
    ropeLocalScalingFactor: 1.0,
    // YARN parameters - only relevant when ropeScalingType='yarn'
    yarnBetaFast: null,
    yarnBetaSlow: null,
    yarnOriginalMaxPos: null,
    // Local YARN parameters - only relevant when ropeLocalScalingType='yarn'
    ropeLocalYarnBetaFast: null,
    ropeLocalYarnBetaSlow: null,
    ropeLocalYarnOriginalMaxPos: null,
  },
  output: {
    finalLogitSoftcapping: null,  // No softcapping (null = disabled)
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
    embeddingPostprocessor: null,
  },
  layerPattern: {
    type: 'uniform',  // All layers same type
    globalPattern: null,  // No alternating pattern (null = not applicable)
    period: null,  // No periodic pattern (null = not applicable)
    offset: null,  // For every_n: first global layer index modulo period
    layerTypes: null,  // For custom: explicit per-layer tags
  },
  chatTemplate: {
    type: null,  // No chat template (null = disabled)
    enabled: false,
  },
  pipeline: null,
  session: null,
  execution: null,
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

  if (manifest.modelType === 'diffusion' || manifest.modelType === 'energy') {
    return;
  }

  const inference = typeof structuredClone === 'function'
    ? structuredClone(manifest.inference)
    : JSON.parse(JSON.stringify(manifest.inference));
  validateRequiredInferenceFields(
    inference,
    manifest.modelId ?? 'unknown'
  );
}

export function hasInferenceConfig(
  manifest
) {
  return manifest.inference != null;
}
