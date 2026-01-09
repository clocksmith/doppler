// =============================================================================
// MoE Routing Config
// =============================================================================

export const DEFAULT_MOE_ROUTING_CONFIG = {
  numExperts: 8,
  topK: 2,
  normalizeWeights: true,
  routerDtype: 'f32',
};

// =============================================================================
// MoE Cache Config
// =============================================================================

export const DEFAULT_MOE_CACHE_CONFIG = {
  dequantCacheMaxEntries: 128,
};

// =============================================================================
// Complete MoE Runtime Config
// =============================================================================

export const DEFAULT_MOE_RUNTIME_CONFIG = {
  routing: DEFAULT_MOE_ROUTING_CONFIG,
  cache: DEFAULT_MOE_CACHE_CONFIG,
};
