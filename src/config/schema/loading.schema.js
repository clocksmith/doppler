import { DEFAULT_DISTRIBUTION_CONFIG } from './distribution.schema.js';
import { DEFAULT_STORAGE_FULL_CONFIG } from './storage.schema.js';
import { MB, GB } from './units.schema.js';

// =============================================================================
// Q4K Layout Default
// =============================================================================

// Default Q4K layout when manifest doesn't specify one (backwards compatibility).
// 'row' = row-wise layout, compatible with fused Q4K kernels
// 'col' = column-wise layout, requires dequantization
export const DEFAULT_Q4K_LAYOUT = 'row';

// =============================================================================
// Shard Cache Config
// =============================================================================

export const DEFAULT_SHARD_CACHE_CONFIG = {
  opfsEntries: 2,
  networkEntries: 16,
  moeMaxEntries: 16,
  verifyHashes: true,
  maxConcurrentLoads: 0,
};

// =============================================================================
// Memory Management Config
// =============================================================================

export const DEFAULT_MEMORY_MANAGEMENT_CONFIG = {
  flushIntervalLayers: 4,
  flushThresholdBytes: 256 * MB,
  gpuQueueFlushLayers: 4,
  logIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// Prefetch Config
// =============================================================================

export const DEFAULT_PREFETCH_CONFIG = {
  enabled: true,
  layersAhead: 1,
  maxShards: 8,
};

// =============================================================================
// OPFS Path Config
// =============================================================================

export const DEFAULT_OPFS_PATH_CONFIG = {
  opfsRootDir: 'doppler-models',
};

// =============================================================================
// Expert Cache Config
// =============================================================================

export const DEFAULT_EXPERT_CACHE_CONFIG = {
  defaultSizeBytes: 2 * GB,
  maxBufferPercentage: 0.25, // 25% of max buffer
  maxBufferFallbackBytes: 256 * MB,
};

// =============================================================================
// Complete Loading Config
// =============================================================================

export const DEFAULT_LOADING_CONFIG = {
  storage: DEFAULT_STORAGE_FULL_CONFIG,
  distribution: DEFAULT_DISTRIBUTION_CONFIG,
  shardCache: DEFAULT_SHARD_CACHE_CONFIG,
  memoryManagement: DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  prefetch: DEFAULT_PREFETCH_CONFIG,
  opfsPath: DEFAULT_OPFS_PATH_CONFIG,
  expertCache: DEFAULT_EXPERT_CACHE_CONFIG,
  allowF32UpcastNonMatmul: false,
};
