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
    storageMode: 'payload-copy',
    tiered: null,
    compactLayerBytes: null,
    compactPrefillBytes: 0,
    compactPrefillReuse: 'none',
    prefillCompactAdmission: 'none',
    sourceReadMode: 'plane-serial',
    expertBatchSize: 1,
    prefillUploadMode: 'expert-copy',
    evictionPolicy: 'lru',
    cacheBytes: 2 * GB,
    hostCacheBytes: 0,
    hostCacheAdmission: 'all',
    hostCacheAdmissionPhase: 'all',
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
  if (!['payload-copy', 'compact-decode', 'tiered-compact'].includes(config.storageMode)) {
    throw new Error('Qwen routed storageMode must be payload-copy, compact-decode or tiered-compact.');
  }
  const compact = config.storageMode === 'compact-decode' || config.storageMode === 'tiered-compact';
  if (config.storageMode === 'tiered-compact') {
    const tiered = config.tiered;
    if (!tiered || typeof tiered !== 'object' || Array.isArray(tiered)
      || Object.keys(tiered).some(key => !['residentLayers', 'parkedLayers', 'mappedCacheBytes', 'mappedProtectedShare'].includes(key))
      || !['residentLayers', 'parkedLayers'].every(key => Array.isArray(tiered[key])
        && tiered[key].every(layer => Number.isSafeInteger(layer) && layer >= 0))
      || new Set([...tiered.residentLayers, ...tiered.parkedLayers]).size !== tiered.residentLayers.length + tiered.parkedLayers.length
      || typeof tiered.mappedProtectedShare !== 'number' || !Number.isFinite(tiered.mappedProtectedShare)
      || tiered.mappedProtectedShare < 0 || tiered.mappedProtectedShare > 1
      || !Number.isSafeInteger(tiered.mappedCacheBytes) || tiered.mappedCacheBytes < 0 || tiered.mappedCacheBytes % 4
      || config.compactPrefillBytes !== 0 || config.compactPrefillReuse !== 'none'
      || config.prefillCompactAdmission !== 'none' || config.sourceReadMode !== 'expert-lease-batch'
      || config.prefillUploadMode !== 'expert-copy') {
      throw new Error('Tiered experts require distinct resident/parked layers, aligned mappedCacheBytes, mappedProtectedShare in [0,1] and batched streaming without full prefill banks.');
    }
  } else if (config.tiered != null) {
    throw new Error('Tiered placement requires storageMode=tiered-compact.');
  }
  if (compact && config.evictionPolicy !== 'lru') {
    throw new Error('Compact decode currently requires evictionPolicy=lru; LFU remains available with payload-copy.');
  }
  if (compact && config.hostCacheBytes !== 0) {
    throw new Error('Compact decode currently requires hostCacheBytes=0.');
  }
  if (!Number.isSafeInteger(config.compactPrefillBytes) || config.compactPrefillBytes < 0
    || config.compactPrefillBytes % 4 || config.compactPrefillBytes >= config.cacheBytes
    || (config.compactPrefillBytes > 0 && config.storageMode !== 'compact-decode')) {
    throw new Error('compactPrefillBytes must be zero or an aligned compact-decode reservation below cacheBytes.');
  }
  if (!['none', 'resident-gpu'].includes(config.compactPrefillReuse)
    || (config.compactPrefillReuse === 'resident-gpu'
      && (config.storageMode !== 'compact-decode' || config.compactPrefillBytes === 0))) {
    throw new Error('compactPrefillReuse requires none or resident-gpu with a compact-decode prefill reservation.');
  }
  if (!['none', 'selected-frequency-first'].includes(config.prefillCompactAdmission)
    || (config.prefillCompactAdmission !== 'none'
      && (config.storageMode !== 'compact-decode' || config.compactPrefillBytes === 0))) {
    throw new Error('prefillCompactAdmission requires none or selected-frequency-first with a compact-decode reservation.');
  }
  if (!['plane-serial', 'expert-parallel', 'lease-parallel', 'expert-lease-batch'].includes(config.sourceReadMode)
    || (config.sourceReadMode !== 'plane-serial'
      && (!compact || config.hostCacheBytes !== 0))) {
    throw new Error('sourceReadMode requires plane-serial, expert-parallel, lease-parallel or expert-lease-batch; parallel modes require compact decode and no host cache.');
  }
  const expertBatchSize = config.expertBatchSize === undefined
    ? DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed.expertBatchSize : config.expertBatchSize;
  if (!Number.isSafeInteger(expertBatchSize) || expertBatchSize < 1 || expertBatchSize > 32
    || (expertBatchSize > 1 && (!compact || config.sourceReadMode !== 'expert-lease-batch'))) {
    throw new Error('expertBatchSize requires 1..32 and batched compact source leases for multiple experts.');
  }
  if (!['expert-copy', 'direct-bank'].includes(config.prefillUploadMode)
    || (config.prefillUploadMode === 'direct-bank'
      && (config.storageMode !== 'compact-decode' || config.hostCacheBytes !== 0))) {
    throw new Error('prefillUploadMode requires expert-copy or direct-bank with compact decode and no host cache.');
  }
  if (config.compactLayerBytes !== null) {
    const entries = Object.entries(config.compactLayerBytes ?? {});
    if (!compact || Array.isArray(config.compactLayerBytes) || !entries.length
      || entries.some(([layer, bytes]) => !/^(0|[1-9][0-9]*)$/.test(layer)
        || !Number.isSafeInteger(Number(layer)) || !Number.isSafeInteger(bytes) || bytes < 4 || bytes % 4)) {
      throw new Error('compactLayerBytes requires null or explicit aligned byte quotas keyed by original layer index in compact-decode mode.');
    }
    const total = entries.reduce((sum, [, bytes]) => sum + bytes, 0);
    if (!Number.isSafeInteger(total) || total > config.cacheBytes - config.compactPrefillBytes) {
      throw new Error('compactLayerBytes exceeds cacheBytes after the prefill reservation.');
    }
  }
  if (!['lru', 'lfu'].includes(config.evictionPolicy)) {
    throw new Error('runtime.loading.expertCache.qwen35Routed.evictionPolicy must be lru or lfu.');
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
  if (!['all', 'decode'].includes(config.hostCacheAdmissionPhase)) {
    throw new Error('runtime.loading.expertCache.qwen35Routed.hostCacheAdmissionPhase must be all or decode.');
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
