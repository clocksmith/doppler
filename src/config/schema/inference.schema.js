// =============================================================================
// Architecture Defaults (Legacy Fallbacks)
// =============================================================================
//
// These are used as fallbacks when manifest.architecture is a string (v1 format).
// For new manifests, architecture should be an object with all fields populated.
// TODO: Remove when v1 manifest support is dropped.

export const DEFAULT_MAX_POSITION_EMBEDDINGS = 8192;
export const DEFAULT_RMS_NORM_EPS = 1e-5;

// =============================================================================
// Layer Pattern Schema
// =============================================================================

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
