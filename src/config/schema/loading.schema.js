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
  rangeCacheBlockBytes: 0,
  rangeCacheMaxBytes: 0,
  rangeCacheMinBytes: 0,
  verificationBufferBytes: 0,
};

export function resolveShardVerificationBufferBytes(value) {
  if (value === undefined) return DEFAULT_SHARD_CACHE_CONFIG.verificationBufferBytes;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error('loading.shardCache.verificationBufferBytes must be a non-negative safe integer.');
  }
  return value;
}

// =============================================================================
// Memory Management Config
// =============================================================================

export const DEFAULT_LOADER_MEMORY_BUDGET_CONFIG = {
  enabled: true,
  maxResidentBytes: null,
  systemMemoryFraction: 0.5,
  reserveBytes: 2 * GB,
  minimumBudgetBytes: 1 * GB,
};

export const DEFAULT_MEMORY_MANAGEMENT_CONFIG = {
  flushIntervalLayers: 4,
  flushThresholdBytes: 256 * MB,
  gpuQueueFlushLayers: 4,
  logIntervalMs: 30000, // 30 seconds
  budget: DEFAULT_LOADER_MEMORY_BUDGET_CONFIG,
};

// =============================================================================
// Prefetch Config
// =============================================================================

export const DEFAULT_PREFETCH_CONFIG = {
  enabled: true,
  layersAhead: 1,
  maxShards: 8,
  allowRangeLoaderPrefetch: false,
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
  evictionHighWatermark: 0.9,
  emergencyTrimToRatio: 0.75,
  gptOssPager: {
    enabled: true,
    maxResidentLayers: 2,
  },
  qwen35Pager: {
    enabled: false,
    maxResidentLayers: 2,
  },
  qwen35Routed: {
    enabled: false,
    cacheBytes: 2 * GB,
    hostCacheBytes: 0,
    hostCacheAdmission: 'all',
    stagingBytes: 16 * MB,
    readConcurrency: 2,
    routeChunkTokens: 256,
    gpuBudgetBytes: 24 * GB,
    reserveBytes: 4 * GB,
  },
};

export function validateQwen35RoutedConfig(config) {
  if (!config || typeof config !== 'object' || typeof config.enabled !== 'boolean') {
    throw new Error('runtime.loading.expertCache.qwen35Routed requires resolved configuration.');
  }
  for (const field of ['cacheBytes', 'stagingBytes', 'readConcurrency', 'routeChunkTokens', 'gpuBudgetBytes', 'reserveBytes']) {
    if (!Number.isSafeInteger(config[field]) || config[field] < 1) {
      throw new Error(`runtime.loading.expertCache.qwen35Routed.${field} must be a positive safe integer.`);
    }
  }
  if (!Number.isSafeInteger(config.hostCacheBytes) || config.hostCacheBytes < 0) {
    throw new Error('runtime.loading.expertCache.qwen35Routed.hostCacheBytes must be a non-negative safe integer.');
  }
  if (!['all', 'repeat-read'].includes(config.hostCacheAdmission)) {
    throw new Error('runtime.loading.expertCache.qwen35Routed.hostCacheAdmission must be all or repeat-read.');
  }
  if (config.stagingBytes < 8 * config.readConcurrency || config.stagingBytes % (8 * config.readConcurrency)
    || config.cacheBytes + config.reserveBytes >= config.gpuBudgetBytes) {
    throw new Error('Qwen routed staging must be 4-byte aligned per reader and leave GPU budget for weights and activations.');
  }
  return config;
}

export function validateQwen35PagerConfig(config) {
  if (!config || typeof config.enabled !== 'boolean'
    || !Number.isSafeInteger(config.maxResidentLayers) || config.maxResidentLayers < 1) {
    throw new Error('runtime.loading.expertCache.qwen35Pager requires enabled and a positive integer maxResidentLayers.');
  }
  return config;
}

// =============================================================================
// Complete Loading Config
// =============================================================================

export const DEFAULT_EXPERT_PREPARATION_CONFIG = {
  mode: 'on-demand',
};

export function validateExpertPreparationConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)
    || !['on-demand', 'eager-resident'].includes(config.mode)) {
    throw new Error('runtime.loading.expertPreparation.mode must be "on-demand" or "eager-resident".');
  }
  return config;
}

export const DEFAULT_LOADING_CONFIG = {
  storage: DEFAULT_STORAGE_FULL_CONFIG,
  distribution: DEFAULT_DISTRIBUTION_CONFIG,
  shardCache: DEFAULT_SHARD_CACHE_CONFIG,
  memoryManagement: DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  prefetch: DEFAULT_PREFETCH_CONFIG,
  opfsPath: DEFAULT_OPFS_PATH_CONFIG,
  expertCache: DEFAULT_EXPERT_CACHE_CONFIG,
  expertPreparation: DEFAULT_EXPERT_PREPARATION_CONFIG,
  allowF32UpcastNonMatmul: false,
};
