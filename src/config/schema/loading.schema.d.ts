/**
 * Loading Config Schema Definitions
 *
 * Configuration for model loading behavior: shard caching, memory management,
 * and storage settings. These values were previously hardcoded across the codebase.
 *
 * @module config/schema/loading
 */

import type { DistributionConfigSchema } from './distribution.schema.js';
import type { StorageFullConfigSchema } from './storage.schema.js';

/**
 * Configuration for the shard LRU cache.
 *
 * The cache stores recently-used model shards to avoid redundant disk/network reads.
 * Different loading scenarios need different cache sizes:
 * - OPFS (disk): Small cache (2 shards) since disk reads are fast
 * - Network: Large cache (16 shards) to avoid re-fetching over network
 * - MoE: Dynamic sizing based on experts per token
 */
export interface ShardCacheConfigSchema {
  /** Max entries when loading from OPFS (disk reads are fast) */
  opfsEntries: number;

  /** Max entries when loading from network (avoid re-fetching) */
  networkEntries: number;

  /** Max entries for MoE models (caps the dynamic formula) */
  moeMaxEntries: number;
}

/** Default shard cache configuration */
export declare const DEFAULT_SHARD_CACHE_CONFIG: ShardCacheConfigSchema;

/**
 * Configuration for memory management during model loading.
 *
 * Controls when to flush caches and GPU queues to manage memory pressure.
 */
export interface MemoryManagementConfigSchema {
  /** Flush shard cache every N layers during loading */
  flushIntervalLayers: number;

  /** Flush shard cache when it exceeds this size in bytes */
  flushThresholdBytes: number;

  /** Flush GPU queue every N layers (releases Chrome staging memory) */
  gpuQueueFlushLayers: number;

  /** Log memory stats every N milliseconds during loading */
  logIntervalMs: number;
}

/** Default memory management configuration */
export declare const DEFAULT_MEMORY_MANAGEMENT_CONFIG: MemoryManagementConfigSchema;

/**
 * Configuration for OPFS directory paths.
 *
 * Note: This is distinct from StorageFullConfigSchema (in storage.schema.js)
 * which handles quota, VRAM estimation, and alignment settings.
 */
export interface OpfsPathConfigSchema {
  /** Root directory name in OPFS for model storage */
  opfsRootDir: string;
}

/** Default OPFS path configuration */
export declare const DEFAULT_OPFS_PATH_CONFIG: OpfsPathConfigSchema;

/**
 * Configuration for the MoE expert LRU cache.
 *
 * Controls how much VRAM is allocated for caching expert weights.
 */
export interface ExpertCacheConfigSchema {
  /** Default maximum cache size in bytes */
  defaultSizeBytes: number;

  /** Maximum percentage of adapter's maxBufferSize to use (0-1) */
  maxBufferPercentage: number;
}

/** Default expert cache configuration */
export declare const DEFAULT_EXPERT_CACHE_CONFIG: ExpertCacheConfigSchema;

/**
 * Complete loading configuration schema.
 *
 * Controls all aspects of model loading behavior.
 */
export interface LoadingConfigSchema {
  /** OPFS quota, VRAM estimation, alignment */
  storage: StorageFullConfigSchema;
  /** Network/download settings */
  distribution: DistributionConfigSchema;
  shardCache: ShardCacheConfigSchema;
  memoryManagement: MemoryManagementConfigSchema;
  opfsPath: OpfsPathConfigSchema;
  expertCache: ExpertCacheConfigSchema;
}

/** Default loading configuration */
export declare const DEFAULT_LOADING_CONFIG: LoadingConfigSchema;
