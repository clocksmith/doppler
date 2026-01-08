/**
 * Shard Cache
 *
 * LRU cache for model shards with deduplication of in-flight requests.
 *
 * @module loader/shard-cache
 */

import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { CustomShardLoader, ShardSourceInfo } from './loader-types.js';
import type { ShardCacheConfigSchema } from '../config/schema/loading.schema.js';

/**
 * Configuration for shard cache
 */
export interface ShardCacheConfig {
  maxEntries: number;
  customLoader?: CustomShardLoader;
  verifyHashes?: boolean;
  manifest?: RDRRManifest;
  /** Loading config for cache sizing */
  loadingConfig?: ShardCacheConfigSchema;
}

/**
 * LRU cache for model shards with request deduplication
 */
export declare class ShardCache {
  /** Track last shard source for progress reporting */
  lastSource: ShardSourceInfo | null;

  constructor(config: ShardCacheConfig);

  /**
   * Update configuration
   */
  configure(config: Partial<ShardCacheConfig>): void;

  /**
   * Set custom shard loader
   */
  setCustomLoader(loader: CustomShardLoader | null, verify?: boolean): void;

  /**
   * Set manifest for hash verification
   */
  setManifest(manifest: RDRRManifest | null): void;

  /**
   * Check if a custom loader is configured
   */
  get hasCustomLoader(): boolean;

  /**
   * Check if shard is cached
   */
  has(shardIndex: number): boolean;

  /**
   * Get cache size
   */
  get size(): number;

  /**
   * Get total cached bytes
   */
  get totalBytes(): number;

  /**
   * Load shard with caching and request deduplication.
   * If the same shard is requested concurrently, all callers wait for the same fetch.
   */
  load(shardIndex: number): Promise<ArrayBuffer>;

  /**
   * Clear the cache
   */
  clear(): void;

  /**
   * Configure cache size based on model type.
   * For MoE models, cache enough shards for 2x top-k experts + 1 dense shard.
   * For dense models, keep the default (configurable via loadingConfig).
   */
  configureForModel(manifest: RDRRManifest | null, hasCustomLoader: boolean): void;
}

/**
 * Create a new shard cache with default settings
 */
export declare function createShardCache(
  maxEntries?: number,
  loadingConfig?: ShardCacheConfigSchema
): ShardCache;
