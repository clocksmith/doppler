/**
 * Loading Config Schema Definitions
 *
 * Configuration for model loading behavior: shard caching, memory management,
 * and storage settings. These values were previously hardcoded across the codebase.
 *
 * @module config/schema/loading
 */

// =============================================================================
// Shard Cache Config
// =============================================================================

/** Default shard cache configuration */
export const DEFAULT_SHARD_CACHE_CONFIG = {
  opfsEntries: 2,
  networkEntries: 16,
  moeMaxEntries: 16,
};

// =============================================================================
// Memory Management Config
// =============================================================================

/** Default memory management configuration */
export const DEFAULT_MEMORY_MANAGEMENT_CONFIG = {
  flushIntervalLayers: 4,
  flushThresholdBytes: 256 * 1024 * 1024, // 256MB
  gpuQueueFlushLayers: 4,
  logIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// OPFS Path Config
// =============================================================================

/** Default OPFS path configuration */
export const DEFAULT_OPFS_PATH_CONFIG = {
  opfsRootDir: 'doppler-models',
};

// =============================================================================
// Expert Cache Config
// =============================================================================

/** Default expert cache configuration */
export const DEFAULT_EXPERT_CACHE_CONFIG = {
  defaultSizeBytes: 2 * 1024 * 1024 * 1024, // 2GB
  maxBufferPercentage: 0.25, // 25% of max buffer
};

// =============================================================================
// Complete Loading Config
// =============================================================================

/** Default loading configuration */
export const DEFAULT_LOADING_CONFIG = {
  shardCache: DEFAULT_SHARD_CACHE_CONFIG,
  memoryManagement: DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  opfsPath: DEFAULT_OPFS_PATH_CONFIG,
  expertCache: DEFAULT_EXPERT_CACHE_CONFIG,
};
