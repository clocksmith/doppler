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

  const patternType = pattern.type ?? null;
  const globalPattern = pattern.globalPattern ?? (patternType === 'every_n' ? 'every_n' : null);

  if (!globalPattern) {
    // Default: all layers are global
    return Array.from({ length: numLayers }, (_, i) => i);
  }

  switch (globalPattern) {
    case 'even':
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % 2 === 0);
    case 'odd':
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % 2 === 1);
    case 'every_n': {
      const n = pattern.period;
      if (!Number.isFinite(n) || n <= 0) {
        throw new Error('layerPattern.period must be a positive number for every_n patterns.');
      }
      return Array.from({ length: numLayers }, (_, i) => i).filter(i => i % n === 0);
    }
    default:
      return Array.from({ length: numLayers }, (_, i) => i);
  }
}
