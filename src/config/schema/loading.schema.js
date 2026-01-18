import { DEFAULT_DISTRIBUTION_CONFIG } from './distribution.schema.js';
import { DEFAULT_STORAGE_FULL_CONFIG } from './storage.schema.js';

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
  flushThresholdBytes: 256 * 1024 * 1024, // 256MB
  gpuQueueFlushLayers: 4,
  logIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// Prefetch Config
// =============================================================================

export const DEFAULT_PREFETCH_CONFIG = {
  enabled: false,
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
  defaultSizeBytes: 2 * 1024 * 1024 * 1024, // 2GB
  maxBufferPercentage: 0.25, // 25% of max buffer
  maxBufferFallbackBytes: 256 * 1024 * 1024, // 256MB
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
