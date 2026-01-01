/**
 * Uniform Buffer Cache
 *
 * Caches small uniform buffers by content hash to avoid repeated allocations.
 * WebLLM-inspired optimization: uniform buffers with identical contents are reused
 * across kernel dispatches instead of being created fresh and destroyed each time.
 */

import { getDevice } from './device.js';
import { DEFAULT_GPU_CACHE_CONFIG } from '../config/schema/gpu-cache.schema.js';

interface UniformCacheEntry {
  buffer: GPUBuffer;
  lastUsed: number;
  refCount: number;
}

interface UniformCacheStats {
  hits: number;
  misses: number;
  evictions: number;
  currentSize: number;
}

/**
 * Fast hash for small ArrayBuffers (uniform buffers are typically 16-64 bytes)
 * Uses FNV-1a variant for speed on small inputs
 */
function hashArrayBuffer(data: ArrayBuffer | SharedArrayBuffer): string {
  const view = new Uint8Array(data);
  let hash = 2166136261; // FNV offset basis

  for (let i = 0; i < view.length; i++) {
    hash ^= view[i];
    hash = Math.imul(hash, 16777619); // FNV prime
  }

  // Convert to hex string for Map key
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Uniform Buffer Cache
 *
 * Provides content-addressed caching for uniform buffers. Buffers with
 * identical contents share the same GPU buffer, reducing allocation overhead.
 */
export class UniformBufferCache {
  private cache: Map<string, UniformCacheEntry> = new Map();
  private stats: UniformCacheStats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    currentSize: 0,
  };

  private readonly maxEntries: number;
  private readonly maxAgeMs: number;

  constructor(
    maxEntries: number = DEFAULT_GPU_CACHE_CONFIG.uniformCacheMaxEntries,
    maxAgeMs: number = DEFAULT_GPU_CACHE_CONFIG.uniformCacheMaxAgeMs
  ) {
    this.maxEntries = maxEntries;
    this.maxAgeMs = maxAgeMs;
  }

  /**
   * Get or create a uniform buffer with the given contents.
   * Returns a cached buffer if one exists with identical data.
   */
  getOrCreate(data: ArrayBuffer | SharedArrayBuffer, label: string): GPUBuffer {
    const hash = hashArrayBuffer(data);
    const existing = this.cache.get(hash);

    if (existing) {
      existing.lastUsed = performance.now();
      existing.refCount++;
      this.stats.hits++;
      return existing.buffer;
    }

    // Cache miss - create new buffer
    this.stats.misses++;

    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not initialized');
    }

    const buffer = device.createBuffer({
      label: `${label}_cached`,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data);

    // Evict if at capacity
    if (this.cache.size >= this.maxEntries) {
      this.evictLRU();
    }

    this.cache.set(hash, {
      buffer,
      lastUsed: performance.now(),
      refCount: 1,
    });
    this.stats.currentSize = this.cache.size;

    return buffer;
  }

  /**
   * Release a reference to a cached buffer.
   * Buffer is NOT destroyed - it stays in cache for reuse.
   * Call this instead of buffer.destroy() for cached uniforms.
   */
  release(buffer: GPUBuffer): void {
    // Find entry by buffer reference
    for (const [hash, entry] of this.cache) {
      if (entry.buffer === buffer) {
        entry.refCount = Math.max(0, entry.refCount - 1);
        return;
      }
    }
    // Buffer not in cache - it may have been created outside the cache
    // Don't destroy it here; caller is responsible
  }

  /**
   * Evict least recently used entry
   */
  private evictLRU(): void {
    let oldestHash: string | null = null;
    let oldestTime = Infinity;

    for (const [hash, entry] of this.cache) {
      // Prefer evicting entries with refCount 0
      if (entry.refCount === 0 && entry.lastUsed < oldestTime) {
        oldestTime = entry.lastUsed;
        oldestHash = hash;
      }
    }

    // If all entries are in use, evict oldest anyway
    if (oldestHash === null) {
      for (const [hash, entry] of this.cache) {
        if (entry.lastUsed < oldestTime) {
          oldestTime = entry.lastUsed;
          oldestHash = hash;
        }
      }
    }

    if (oldestHash) {
      const entry = this.cache.get(oldestHash);
      if (entry) {
        entry.buffer.destroy();
        this.cache.delete(oldestHash);
        this.stats.evictions++;
        this.stats.currentSize = this.cache.size;
      }
    }
  }

  /**
   * Evict stale entries (older than maxAgeMs)
   */
  evictStale(): number {
    const now = performance.now();
    let evicted = 0;

    for (const [hash, entry] of this.cache) {
      if (entry.refCount === 0 && now - entry.lastUsed > this.maxAgeMs) {
        entry.buffer.destroy();
        this.cache.delete(hash);
        evicted++;
      }
    }

    this.stats.evictions += evicted;
    this.stats.currentSize = this.cache.size;
    return evicted;
  }

  /**
   * Clear all cached buffers
   */
  clear(): void {
    for (const entry of this.cache.values()) {
      entry.buffer.destroy();
    }
    this.cache.clear();
    this.stats.currentSize = 0;
  }

  /**
   * Check if a buffer is managed by this cache
   */
  isCached(buffer: GPUBuffer): boolean {
    for (const entry of this.cache.values()) {
      if (entry.buffer === buffer) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get cache statistics
   */
  getStats(): UniformCacheStats & { hitRate: string } {
    const total = this.stats.hits + this.stats.misses;
    const hitRate = total > 0 ? ((this.stats.hits / total) * 100).toFixed(1) + '%' : '0%';
    return { ...this.stats, hitRate };
  }
}

/**
 * Release or destroy a uniform buffer appropriately.
 * If the buffer is cached, releases it back to the cache.
 * If not cached, destroys it directly.
 */
export function releaseUniformBuffer(buffer: GPUBuffer): void {
  const cache = getUniformCache();
  if (cache.isCached(buffer)) {
    cache.release(buffer);
  } else {
    buffer.destroy();
  }
}

// Global singleton instance
let globalUniformCache: UniformBufferCache | null = null;

/**
 * Get the global uniform buffer cache instance
 */
export function getUniformCache(): UniformBufferCache {
  if (!globalUniformCache) {
    globalUniformCache = new UniformBufferCache();
  }
  return globalUniformCache;
}

/**
 * Reset the global uniform cache (useful for testing or device loss)
 */
export function resetUniformCache(): void {
  if (globalUniformCache) {
    globalUniformCache.clear();
    globalUniformCache = null;
  }
}
