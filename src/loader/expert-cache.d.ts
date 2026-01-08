/**
 * Expert LRU Cache for MoE Models
 *
 * Tracks expert residency in VRAM and implements LRU eviction
 * to manage memory pressure during inference.
 *
 * @module loader/expert-cache
 */

import type { ExpertWeights } from './weights.js';
import type { ExpertCacheConfigSchema } from '../config/schema/loading.schema.js';

/**
 * Cache statistics
 */
export interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  currentSize: number;
  maxSize: number;
  expertCount: number;
  hitRate: number;
  inUseCount: number;
  pinnedCount: number;
}

/**
 * Expert LRU Cache
 *
 * Manages expert weight residency in VRAM with LRU eviction policy.
 */
export declare class ExpertCache {
  /**
   * Create expert cache
   * @param maxBytes Maximum cache size in bytes (uses config default if not specified)
   * @param config Expert cache configuration
   */
  constructor(maxBytes?: number, config?: ExpertCacheConfigSchema);

  /**
   * Update cache configuration at runtime.
   */
  configure(config: ExpertCacheConfigSchema, maxBytes?: number): void;

  /**
   * Auto-tune cache size based on available VRAM
   * Call this after WebGPU is initialized
   */
  autoTune(): Promise<void>;

  /**
   * Get expert from cache
   * @returns Expert weights or null if not in cache
   */
  get(layerIdx: number, expertIdx: number): ExpertWeights | null;

  /**
   * Put expert into cache
   * @param weights Expert weights to cache
   * @param sizeBytes Size of expert in bytes (for memory tracking)
   */
  put(layerIdx: number, expertIdx: number, weights: ExpertWeights, sizeBytes: number): void;

  /**
   * Check if expert is in cache
   */
  has(layerIdx: number, expertIdx: number): boolean;

  /**
   * Evict least recently used expert
   * Skips experts that are in-use or pinned
   * @returns true if an expert was evicted, false if all experts are protected
   */
  evictLRU(): boolean;

  /**
   * Mark expert as in-use (prevents eviction during inference)
   */
  markInUse(layerIdx: number, expertIdx: number): void;

  /**
   * Mark expert as no longer in use (allows eviction)
   */
  markNotInUse(layerIdx: number, expertIdx: number): void;

  /**
   * Clear all in-use markers (call after inference completes)
   */
  clearInUse(): void;

  /**
   * Pin expert (prevents eviction, for shared experts)
   */
  pinExpert(layerIdx: number, expertIdx: number): void;

  /**
   * Unpin expert (allows eviction)
   */
  unpinExpert(layerIdx: number, expertIdx: number): void;

  /**
   * Pin all shared experts for a model
   */
  pinSharedExperts(sharedExpertIndices: number[], numLayers: number): void;

  /**
   * Check if expert is pinned
   */
  isPinned(layerIdx: number, expertIdx: number): boolean;

  /**
   * Get current memory usage in bytes
   */
  getMemoryUsage(): number;

  /**
   * Get cache statistics
   */
  getStats(): CacheStats;

  /**
   * Clear all cached experts
   */
  clear(): void;

  /**
   * Set maximum cache size
   * @param maxBytes New maximum size in bytes
   */
  setMaxSize(maxBytes: number): void;

  /**
   * Prefetch experts (hint for future access)
   * This is a no-op in the cache - actual prefetch happens in the loader
   */
  prefetch(_layerIdx: number, _expertIndices: number[]): void;

  /**
   * Get all cached expert keys
   */
  getCachedExperts(): Array<{ layerIdx: number; expertIdx: number }>;
}

/**
 * Get global expert cache instance
 */
export declare function getExpertCache(config?: ExpertCacheConfigSchema): ExpertCache;

/**
 * Create new expert cache with custom size
 */
export declare function createExpertCache(maxBytes?: number, config?: ExpertCacheConfigSchema): ExpertCache;

export default ExpertCache;
