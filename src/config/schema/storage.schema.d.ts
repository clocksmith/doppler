/**
 * Storage Config Schema
 *
 * Configuration for OPFS storage, quota management, and memory estimation.
 * These settings control how the system monitors disk space, estimates VRAM,
 * and aligns storage buffers.
 *
 * @module config/schema/storage
 */

/**
 * Configuration for OPFS quota monitoring.
 *
 * Controls thresholds for warning about low disk space and
 * the frequency of quota checks.
 */
export interface QuotaConfigSchema {
  /** Threshold in bytes below which space is considered low (warn user) */
  lowSpaceThresholdBytes: number;

  /** Threshold in bytes below which space is critical (block operations) */
  criticalSpaceThresholdBytes: number;

  /** Interval in milliseconds between quota monitoring checks */
  monitorIntervalMs: number;
}

/** Default quota configuration */
export declare const DEFAULT_QUOTA_CONFIG: QuotaConfigSchema;

/**
 * Configuration for VRAM estimation on different platforms.
 *
 * Used to estimate available GPU memory when WebGPU doesn't provide
 * accurate limits (especially on unified memory systems like Apple Silicon).
 */
export interface VramEstimationConfigSchema {
  /** Ratio of system RAM to use for VRAM estimation on unified memory systems (0-1) */
  unifiedMemoryRatio: number;

  /** Fallback VRAM size in bytes when estimation is not possible */
  fallbackVramBytes: number;

  /** Headroom to leave when VRAM is low, in bytes */
  lowVramHeadroomBytes: number;
}

/** Default VRAM estimation configuration */
export declare const DEFAULT_VRAM_ESTIMATION_CONFIG: VramEstimationConfigSchema;

/**
 * Configuration for storage buffer alignment.
 *
 * Ensures buffers are aligned to optimal boundaries for GPU access.
 */
export interface StorageAlignmentConfigSchema {
  /** Alignment boundary in bytes for storage buffers */
  bufferAlignmentBytes: number;
}

/** Default storage alignment configuration */
export declare const DEFAULT_STORAGE_ALIGNMENT_CONFIG: StorageAlignmentConfigSchema;

export type StorageBackendMode = 'auto' | 'opfs' | 'indexeddb' | 'memory';

export interface OpfsBackendConfigSchema {
  /** Use SyncAccessHandle when available (worker-only) */
  useSyncAccessHandle: boolean;
  /** Maximum concurrent OPFS handles */
  maxConcurrentHandles: number;
}

export interface IndexeddbBackendConfigSchema {
  /** IndexedDB database name */
  dbName: string;
  /** Object store for shard chunks */
  shardStore: string;
  /** Object store for manifest/tokenizer/meta */
  metaStore: string;
  /** Chunk size in bytes for shard storage */
  chunkSizeBytes: number;
}

export interface MemoryBackendConfigSchema {
  /** Max in-memory bytes for fallback storage */
  maxBytes: number;
}

export interface StorageStreamingConfigSchema {
  /** Target read chunk size in bytes */
  readChunkBytes: number;
  /** Maximum in-flight bytes across readers */
  maxInFlightBytes: number;
  /** Use BYOB readers when supported */
  useByob: boolean;
}

export interface StorageBackendConfigSchema {
  /** Requested backend (auto selects best available) */
  backend: StorageBackendMode;
  opfs: OpfsBackendConfigSchema;
  indexeddb: IndexeddbBackendConfigSchema;
  memory: MemoryBackendConfigSchema;
  streaming: StorageStreamingConfigSchema;
}

/** Default backend configuration */
export declare const DEFAULT_STORAGE_BACKEND_CONFIG: StorageBackendConfigSchema;

/**
 * Complete storage configuration schema.
 *
 * Combines quota monitoring, VRAM estimation, and alignment settings.
 */
export interface StorageFullConfigSchema {
  quota: QuotaConfigSchema;
  vramEstimation: VramEstimationConfigSchema;
  alignment: StorageAlignmentConfigSchema;
  backend: StorageBackendConfigSchema;
}

/** Default storage configuration */
export declare const DEFAULT_STORAGE_FULL_CONFIG: StorageFullConfigSchema;
