/**
 * Expert LRU Cache for MoE Models
 *
 * Tracks expert residency in VRAM and implements LRU eviction
 * to manage memory pressure during inference.
 *
 * @module loader/expert-cache
 */

import { releaseBuffer } from '../gpu/buffer-pool.js';
import type { ExpertWeights } from './doppler-loader.js';

/**
 * Cache entry with access tracking
 */
interface CacheEntry {
  weights: ExpertWeights;
  lastAccess: number;
  sizeBytes: number;
}

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
export class ExpertCache {
  private cache = new Map<string, CacheEntry>();
  private maxBytes: number;
  private currentBytes = 0;
  private accessCounter = 0;

  // Statistics
  private hits = 0;
  private misses = 0;
  private evictions = 0;

  // Smart eviction: track experts currently in use (don't evict)
  private inUse = new Set<string>();

  // Pinned experts: shared experts that should never be evicted
  private pinned = new Set<string>();

  /**
   * Create expert cache
   * @param maxBytes Maximum cache size in bytes (default: 2GB)
   */
  constructor(maxBytes: number = 2 * 1024 * 1024 * 1024) {
    this.maxBytes = maxBytes;
  }

  /**
   * Auto-tune cache size based on available VRAM
   * Call this after WebGPU is initialized
   */
  async autoTune(): Promise<void> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      console.log('[ExpertCache] WebGPU not available, using default 2GB');
      return;
    }

    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (!adapter) {
        console.log('[ExpertCache] No GPU adapter, using default 2GB');
        return;
      }

      const limits = adapter.limits;
      const maxBufferSize = limits?.maxBufferSize || 256 * 1024 * 1024;

      // Heuristic: Use up to 2GB or 25% of max buffer size, whichever is smaller
      // This leaves room for model weights, KV cache, and activations
      const autoSize = Math.min(
        2 * 1024 * 1024 * 1024, // 2GB cap
        Math.floor(maxBufferSize * 0.25)
      );

      this.maxBytes = autoSize;
      console.log(`[ExpertCache] Auto-tuned to ${(this.maxBytes / 1024 / 1024).toFixed(0)}MB (maxBuffer: ${(maxBufferSize / 1024 / 1024).toFixed(0)}MB)`);
    } catch (e) {
      console.warn('[ExpertCache] Auto-tune failed, using default 2GB:', e);
    }
  }

  /**
   * Generate cache key for expert
   */
  private getKey(layerIdx: number, expertIdx: number): string {
    return `${layerIdx}_${expertIdx}`;
  }

  /**
   * Get expert from cache
   * @returns Expert weights or null if not in cache
   */
  get(layerIdx: number, expertIdx: number): ExpertWeights | null {
    const key = this.getKey(layerIdx, expertIdx);
    const entry = this.cache.get(key);

    if (entry) {
      // Update access time for LRU tracking
      entry.lastAccess = ++this.accessCounter;
      this.hits++;
      return entry.weights;
    }

    this.misses++;
    return null;
  }

  /**
   * Put expert into cache
   * @param weights Expert weights to cache
   * @param sizeBytes Size of expert in bytes (for memory tracking)
   */
  put(layerIdx: number, expertIdx: number, weights: ExpertWeights, sizeBytes: number): void {
    const key = this.getKey(layerIdx, expertIdx);

    // If already in cache, update it
    if (this.cache.has(key)) {
      const existing = this.cache.get(key)!;
      this.currentBytes -= existing.sizeBytes;
    }

    // Evict entries if needed to make room
    while (this.currentBytes + sizeBytes > this.maxBytes && this.cache.size > 0) {
      this.evictLRU();
    }

    // Add to cache
    this.cache.set(key, {
      weights,
      lastAccess: ++this.accessCounter,
      sizeBytes,
    });
    this.currentBytes += sizeBytes;
  }

  /**
   * Check if expert is in cache
   */
  has(layerIdx: number, expertIdx: number): boolean {
    return this.cache.has(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Evict least recently used expert
   * Skips experts that are in-use or pinned
   * @returns true if an expert was evicted, false if all experts are protected
   */
  evictLRU(): boolean {
    if (this.cache.size === 0) return false;

    let lruKey: string | null = null;
    let lruTime = Infinity;

    for (const [key, entry] of this.cache) {
      // Skip in-use experts (currently being used in inference)
      if (this.inUse.has(key)) continue;
      // Skip pinned experts (shared experts that should never be evicted)
      if (this.pinned.has(key)) continue;

      if (entry.lastAccess < lruTime) {
        lruTime = entry.lastAccess;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.evict(lruKey);
      return true;
    }

    // All experts are either in-use or pinned
    return false;
  }

  /**
   * Mark expert as in-use (prevents eviction during inference)
   */
  markInUse(layerIdx: number, expertIdx: number): void {
    this.inUse.add(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Mark expert as no longer in use (allows eviction)
   */
  markNotInUse(layerIdx: number, expertIdx: number): void {
    this.inUse.delete(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Clear all in-use markers (call after inference completes)
   */
  clearInUse(): void {
    this.inUse.clear();
  }

  /**
   * Pin expert (prevents eviction, for shared experts)
   */
  pinExpert(layerIdx: number, expertIdx: number): void {
    this.pinned.add(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Unpin expert (allows eviction)
   */
  unpinExpert(layerIdx: number, expertIdx: number): void {
    this.pinned.delete(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Pin all shared experts for a model
   */
  pinSharedExperts(sharedExpertIndices: number[], numLayers: number): void {
    for (let layer = 0; layer < numLayers; layer++) {
      for (const expertIdx of sharedExpertIndices) {
        this.pinExpert(layer, expertIdx);
      }
    }
    console.log(`[ExpertCache] Pinned ${sharedExpertIndices.length} shared experts across ${numLayers} layers`);
  }

  /**
   * Check if expert is pinned
   */
  isPinned(layerIdx: number, expertIdx: number): boolean {
    return this.pinned.has(this.getKey(layerIdx, expertIdx));
  }

  /**
   * Evict specific expert by key
   */
  private evict(key: string): void {
    const entry = this.cache.get(key);
    if (!entry) return;

    // Release GPU buffers
    this.releaseExpertBuffers(entry.weights);

    this.currentBytes -= entry.sizeBytes;
    this.cache.delete(key);
    this.evictions++;

    console.log(`[ExpertCache] Evicted expert ${key}, freed ${(entry.sizeBytes / 1024 / 1024).toFixed(1)}MB`);
  }

  /**
   * Release GPU buffers for expert weights
   */
  private releaseExpertBuffers(weights: ExpertWeights): void {
    const buffers = [
      weights.gate,
      weights.up,
      weights.down,
      weights.gateUpBlocks,
      weights.gateUpScales,
      weights.gateUpBias,
      weights.downBlocks,
      weights.downScales,
      weights.downBias,
    ];

    for (const buf of buffers) {
      if (buf instanceof GPUBuffer) {
        try {
          releaseBuffer(buf);
        } catch (e) {
          // Buffer may already be released
        }
      }
    }
  }

  /**
   * Get current memory usage in bytes
   */
  getMemoryUsage(): number {
    return this.currentBytes;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      currentSize: this.currentBytes,
      maxSize: this.maxBytes,
      expertCount: this.cache.size,
      hitRate: total > 0 ? this.hits / total : 0,
      inUseCount: this.inUse.size,
      pinnedCount: this.pinned.size,
    };
  }

  /**
   * Clear all cached experts
   */
  clear(): void {
    for (const [, entry] of this.cache) {
      this.releaseExpertBuffers(entry.weights);
    }
    this.cache.clear();
    this.currentBytes = 0;
    this.inUse.clear();
    // Note: pinned is NOT cleared - shared experts stay pinned
    console.log('[ExpertCache] Cache cleared');
  }

  /**
   * Set maximum cache size
   * @param maxBytes New maximum size in bytes
   */
  setMaxSize(maxBytes: number): void {
    this.maxBytes = maxBytes;

    // Evict if over new limit
    while (this.currentBytes > this.maxBytes && this.cache.size > 0) {
      this.evictLRU();
    }
  }

  /**
   * Prefetch experts (hint for future access)
   * This is a no-op in the cache - actual prefetch happens in the loader
   */
  prefetch(_layerIdx: number, _expertIndices: number[]): void {
    // Prefetch hint - the loader should implement actual prefetch logic
  }

  /**
   * Get all cached expert keys
   */
  getCachedExperts(): Array<{ layerIdx: number; expertIdx: number }> {
    const result: Array<{ layerIdx: number; expertIdx: number }> = [];
    for (const key of this.cache.keys()) {
      const [layer, expert] = key.split('_').map(Number);
      result.push({ layerIdx: layer, expertIdx: expert });
    }
    return result;
  }
}

// Global cache instance
let globalCache: ExpertCache | null = null;

/**
 * Get global expert cache instance
 */
export function getExpertCache(): ExpertCache {
  if (!globalCache) {
    globalCache = new ExpertCache();
  }
  return globalCache;
}

/**
 * Create new expert cache with custom size
 */
export function createExpertCache(maxBytes?: number): ExpertCache {
  return new ExpertCache(maxBytes);
}

export default ExpertCache;
