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

/**
 * Configuration for the uniform buffer cache.
 *
 * The uniform buffer cache stores small GPU buffers by content hash,
 * allowing reuse across kernel dispatches instead of repeated allocations.
 */
export interface GpuCacheConfigSchema {
  /** Maximum number of entries in the uniform buffer cache */
  uniformCacheMaxEntries: number;

  /** Maximum age in milliseconds before an unused entry becomes stale */
  uniformCacheMaxAgeMs: number;
}

/** Default GPU cache configuration */
export const DEFAULT_GPU_CACHE_CONFIG: GpuCacheConfigSchema = {
  uniformCacheMaxEntries: 256,
  uniformCacheMaxAgeMs: 60000, // 60 seconds
};
