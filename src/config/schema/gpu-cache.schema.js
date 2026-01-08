/**
 * GPU Cache Config Schema
 *
 * Configuration for GPU uniform buffer caching.
 * These settings control cache size, entry limits, and expiration policies
 * for the uniform buffer cache that reduces GPU buffer allocations.
 *
 * @module config/schema/gpu-cache
 */

// =============================================================================
// Uniform Buffer Cache Config
// =============================================================================

/** Default GPU cache configuration */
export const DEFAULT_GPU_CACHE_CONFIG = {
  uniformCacheMaxEntries: 256,
  uniformCacheMaxAgeMs: 60000, // 60 seconds
};
