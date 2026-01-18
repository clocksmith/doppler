// =============================================================================
// Quota Config
// =============================================================================

export const DEFAULT_QUOTA_CONFIG = {
  lowSpaceThresholdBytes: 500 * 1024 * 1024, // 500MB
  criticalSpaceThresholdBytes: 100 * 1024 * 1024, // 100MB
  monitorIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// VRAM Estimation Config
// =============================================================================

export const DEFAULT_VRAM_ESTIMATION_CONFIG = {
  unifiedMemoryRatio: 0.5, // 50% of system RAM
  fallbackVramBytes: 2 * 1024 * 1024 * 1024, // 2GB
  lowVramHeadroomBytes: 500 * 1024 * 1024, // 500MB
};

// =============================================================================
// Storage Alignment Config
// =============================================================================

export const DEFAULT_STORAGE_ALIGNMENT_CONFIG = {
  bufferAlignmentBytes: 4096, // 4KB alignment (typical page size)
};

// =============================================================================
// Storage Backend Config
// =============================================================================

export const DEFAULT_STORAGE_BACKEND_CONFIG = {
  backend: 'auto', // auto | opfs | indexeddb | memory
  opfs: {
    useSyncAccessHandle: true,
    maxConcurrentHandles: 2,
  },
  indexeddb: {
    dbName: 'doppler-models',
    shardStore: 'shards',
    metaStore: 'meta',
    chunkSizeBytes: 4 * 1024 * 1024, // 4MB chunks
  },
  memory: {
    maxBytes: 512 * 1024 * 1024, // 512MB cap for non-persistent fallback
  },
  streaming: {
    readChunkBytes: 4 * 1024 * 1024, // 4MB
    maxInFlightBytes: 64 * 1024 * 1024, // 64MB
    useByob: true,
  },
};

// =============================================================================
// Complete Storage Config
// =============================================================================

export const DEFAULT_STORAGE_FULL_CONFIG = {
  quota: DEFAULT_QUOTA_CONFIG,
  vramEstimation: DEFAULT_VRAM_ESTIMATION_CONFIG,
  alignment: DEFAULT_STORAGE_ALIGNMENT_CONFIG,
  backend: DEFAULT_STORAGE_BACKEND_CONFIG,
};
