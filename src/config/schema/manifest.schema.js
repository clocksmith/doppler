// =============================================================================
// Hash & Versioning
// =============================================================================

export const RDRR_VERSION = 1;

export const SHARD_SIZE = 64 * 1024 * 1024;

export const TENSORS_FILENAME = 'tensors.json';

// =============================================================================
// Inference Schema (Model-Specific Inference Parameters)
// =============================================================================

export const DEFAULT_MANIFEST_INFERENCE = {
  attention: {
    queryPreAttnScalar: 8,  // sqrt(64) for standard 64-dim heads
    attnLogitSoftcapping: null,  // No softcapping (null = disabled)
    slidingWindow: null,  // Full attention (null = no sliding window)
    queryKeyNorm: false,
  },
  normalization: {
    rmsNormEps: 1e-5,
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
