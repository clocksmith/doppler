/**
 * MoE Config Schema
 *
 * Configuration for Mixture of Experts routing and caching.
 * These are runtime settings that affect how MoE layers execute.
 *
 * @module config/schema/moe
 */

// =============================================================================
// Router Dtype
// =============================================================================

/** Supported data types for router computation */
export type RouterDtype = 'f16' | 'f32';

// =============================================================================
// MoE Routing Config
// =============================================================================

/**
 * Configuration for MoE routing behavior.
 *
 * Controls how tokens are routed to experts during inference.
 */
export interface MoERoutingConfigSchema {
  /** Number of experts in the MoE layer */
  numExperts: number;

  /** Number of experts to activate per token (top-K routing) */
  topK: number;

  /** Normalize expert weights after routing */
  normalizeWeights: boolean;

  /** Data type for router computation */
  routerDtype: RouterDtype;
}

/** Default MoE routing configuration */
export const DEFAULT_MOE_ROUTING_CONFIG: MoERoutingConfigSchema = {
  numExperts: 8,
  topK: 2,
  normalizeWeights: true,
  routerDtype: 'f32',
};

// =============================================================================
// MoE Cache Config
// =============================================================================

/**
 * Configuration for MoE dequantization caching.
 *
 * Controls the LRU cache for dequantized expert weights to avoid
 * redundant dequantization when experts are reused across tokens.
 */
export interface MoECacheConfigSchema {
  /** Maximum number of dequantized expert entries to cache */
  dequantCacheMaxEntries: number;
}

/** Default MoE cache configuration */
export const DEFAULT_MOE_CACHE_CONFIG: MoECacheConfigSchema = {
  dequantCacheMaxEntries: 128,
};

// =============================================================================
// Complete MoE Runtime Config
// =============================================================================

/**
 * Complete MoE runtime configuration schema.
 *
 * Controls all aspects of MoE inference behavior.
 */
export interface MoERuntimeConfigSchema {
  routing: MoERoutingConfigSchema;
  cache: MoECacheConfigSchema;
}

/** Default MoE runtime configuration */
export const DEFAULT_MOE_RUNTIME_CONFIG: MoERuntimeConfigSchema = {
  routing: DEFAULT_MOE_ROUTING_CONFIG,
  cache: DEFAULT_MOE_CACHE_CONFIG,
};
