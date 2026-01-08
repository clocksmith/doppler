/**
 * MoE Config Schema
 *
 * Configuration for Mixture of Experts routing and caching.
 * These are runtime settings that affect how MoE layers execute.
 *
 * @module config/schema/moe
 */

// =============================================================================
// MoE Routing Config
// =============================================================================

/** Default MoE routing configuration */
export const DEFAULT_MOE_ROUTING_CONFIG = {
  numExperts: 8,
  topK: 2,
  normalizeWeights: true,
  routerDtype: 'f32',
};

// =============================================================================
// MoE Cache Config
// =============================================================================

/** Default MoE cache configuration */
export const DEFAULT_MOE_CACHE_CONFIG = {
  dequantCacheMaxEntries: 128,
};

// =============================================================================
// Complete MoE Runtime Config
// =============================================================================

/** Default MoE runtime configuration */
export const DEFAULT_MOE_RUNTIME_CONFIG = {
  routing: DEFAULT_MOE_ROUTING_CONFIG,
  cache: DEFAULT_MOE_CACHE_CONFIG,
};
