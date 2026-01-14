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
