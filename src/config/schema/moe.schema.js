// =============================================================================
// MoE Routing Config
// =============================================================================

export const DEFAULT_MOE_ROUTING_CONFIG = {
  numExperts: 8,
  topK: 2,
  normalizeWeights: true,
  routerDtype: 'f32',
  maxTokensPerExpert: 0,
  maxTokensPerExpertHeadroom: 2.0,
  maxTokensPerExpertMin: 4,
  maxTokensPerExpertCap: 0,
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

export function validateMoeRoutingConfig(routing) {
  if (!routing || typeof routing !== 'object') return;
  if (typeof routing.numExperts === 'number' && (routing.numExperts <= 0 || !Number.isInteger(routing.numExperts))) {
    throw new Error(
      `DopplerConfigError: moe.routing.numExperts must be a positive integer; got ${routing.numExperts}.`
    );
  }
}
