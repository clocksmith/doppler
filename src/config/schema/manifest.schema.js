/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

// =============================================================================
// Hash & Versioning
// =============================================================================

/** RDRR format version */
export const RDRR_VERSION = 1;

/** Default shard size (64MB) */
export const SHARD_SIZE = 64 * 1024 * 1024;

/** External tensors filename */
export const TENSORS_FILENAME = 'tensors.json';

// =============================================================================
// Inference Schema (Model-Specific Inference Parameters)
// =============================================================================

/**
 * Standard inference configuration template.
 *
 * PURPOSE: Converter template and test fixtures ONLY.
 * NOT a runtime fallback - if manifest is missing fields, validation fails.
 *
 * These values represent a "standard transformer" (no special features).
 * Converter uses this as a base, then overrides for specific model families.
 */
export const DEFAULT_MANIFEST_INFERENCE = {
  attention: {
    queryPreAttnScalar: 8,  // sqrt(64) for standard 64-dim heads
    attnLogitSoftcapping: null,  // No softcapping
    slidingWindow: null,  // Full attention
    queryKeyNorm: false,
  },
  normalization: {
    rmsNormWeightOffset: false,
    postAttentionNorm: false,
    preFeedforwardNorm: false,
    postFeedforwardNorm: false,
  },
  ffn: {
    activation: 'silu',
    gatedActivation: true,
  },
  rope: {
    ropeTheta: 10000,
    ropeLocalTheta: null,  // Same as ropeTheta
    ropeScalingType: null,  // No scaling
    ropeScalingFactor: 1.0,
    yarnBetaFast: null,  // No YARN
    yarnBetaSlow: null,
    yarnOriginalMaxPos: null,
  },
  output: {
    finalLogitSoftcapping: null,  // No softcapping
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
  },
};

// =============================================================================
// Validation Helpers
// =============================================================================

/** Check if manifest is v1 format (has groups) */
export function isV1Manifest(manifest) {
  return manifest.version === 1 && !!manifest.groups;
}

/** Check if manifest has MoE config */
export function hasMoEConfig(manifest) {
  return manifest.moeConfig != null && manifest.moeConfig.numExperts > 1;
}

/**
 * Validate manifest has required inference configuration.
 * Throws if manifest is missing inference field (legacy manifest).
 *
 * @throws Error if manifest.inference is missing
 */
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

/**
 * Type guard to check if manifest has inference config.
 * Use validateManifestInference() to fail fast; this is for conditional checks.
 */
export function hasInferenceConfig(
  manifest
) {
  return manifest.inference != null;
}
