/**
 * Inference Schema Definitions
 *
 * Configuration for model inference behavior.
 * These are runtime settings that affect how the model executes.
 *
 * @module config/schema/inference
 */

// =============================================================================
// RoPE (Rotary Position Embedding)
// =============================================================================

/** Default RoPE configuration */
export const DEFAULT_ROPE_CONFIG = {
  ropeTheta: 10000,
  ropeLocalTheta: undefined,
  ropeScalingType: null,
  ropeScalingFactor: 1.0,
  yarnBetaFast: 32,
  yarnBetaSlow: 1,
  yarnOriginalMaxPos: 4096,
};

// =============================================================================
// Architecture Defaults
// =============================================================================

/**
 * Default max position embeddings when not specified in model config.
 * Used as fallback when parsing incomplete manifests/configs.
 * Modern models typically support 8192+, so this is a conservative default.
 */
export const DEFAULT_MAX_POSITION_EMBEDDINGS = 8192;

// =============================================================================
// Layer Pattern Schema
// =============================================================================

/**
 * Compute global attention layer indices from pattern.
 * Used at runtime when numLayers is known.
 */
export function computeGlobalLayers(
  pattern,
  numLayers
) {
  if (pattern.attentionLayers) {
    // Legacy: use explicit array (filtered to valid range)
    return pattern.attentionLayers.filter(i => i < numLayers);
  }

  if (!pattern.globalPattern) {
    // Default: all layers are global
    return Array.from({ length: numLayers }, (_, i) => i);
  }

  switch (pattern.globalPattern) {
    case 'even':
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % 2 === 0);
    case 'odd':
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % 2 === 1);
    case 'every_n': {
      const n = pattern.globalPatternN ?? 6;
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % n === 0);
    }
    default:
      return Array.from({ length: numLayers }, (_, i) => i);
  }
}
