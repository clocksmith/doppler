/**
 * Uniform Buffer Cache
 *
 * Caches small uniform buffers by content hash to avoid repeated allocations.
 * WebLLM-inspired optimization: uniform buffers with identical contents are reused
 * across kernel dispatches instead of being created fresh and destroyed each time.
 */

import { getDevice } from './device.js';
import { getRuntimeConfig } from '../config/runtime.js';

/**
 * Fast hash for small ArrayBuffers (uniform buffers are typically 16-64 bytes)
 * Uses FNV-1a variant for speed on small inputs
 * @param {ArrayBuffer | SharedArrayBuffer} data
 * @returns {string}
 */
function hashArrayBuffer(data) {
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
 *
 * IMPORTANT: Evicted buffers are NOT destroyed immediately. They are queued
 * for deferred destruction to avoid use-after-destroy bugs when command
 * buffers reference cached uniforms that get evicted before submit.
 * Call flushPendingDestruction() after GPU work completes.
 */
export class UniformBufferCache {
  /** @type {Map<string, {buffer: GPUBuffer, lastUsed: number, refCount: number}>} */
  #cache = new Map();

  /** @type {{hits: number, misses: number, evictions: number, currentSize: number}} */
  #stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    currentSize: 0,
  };

  /** @type {GPUBuffer[]} Buffers evicted from cache, awaiting destruction after GPU work completes */
  #pendingDestruction = [];

  /** @type {number} */
  #maxEntries;

  /** @type {number} */
  #maxAgeMs;

  /**
   * @param {number} [maxEntries]
   * @param {number} [maxAgeMs]
   */
  constructor(
    maxEntries = getRuntimeConfig().shared.gpuCache.uniformCacheMaxEntries,
    maxAgeMs = getRuntimeConfig().shared.gpuCache.uniformCacheMaxAgeMs
  ) {
    this.#maxEntries = maxEntries;
    this.#maxAgeMs = maxAgeMs;
  }

  /**
   * Get or create a uniform buffer with the given contents.
   * Returns a cached buffer if one exists with identical data.
   * @param {ArrayBuffer | SharedArrayBuffer} data
   * @param {string} label
   * @returns {GPUBuffer}
   */
  getOrCreate(data, label) {
    const hash = hashArrayBuffer(data);
    const existing = this.#cache.get(hash);

    if (existing) {
      existing.lastUsed = performance.now();
      existing.refCount++;
      this.#stats.hits++;
      return existing.buffer;
    }

    // Cache miss - create new buffer
    this.#stats.misses++;

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
    if (this.#cache.size >= this.#maxEntries) {
      this.#evictLRU();
    }

    this.#cache.set(hash, {
      buffer,
      lastUsed: performance.now(),
      refCount: 1,
    });
    this.#stats.currentSize = this.#cache.size;

    return buffer;
  }

  /**
   * Release a reference to a cached buffer.
   * Buffer is NOT destroyed - it stays in cache for reuse.
   * Call this instead of buffer.destroy() for cached uniforms.
   * @param {GPUBuffer} buffer
   * @returns {void}
   */
  release(buffer) {
    // Find entry by buffer reference
    for (const [hash, entry] of this.#cache) {
      if (entry.buffer === buffer) {
        entry.refCount = Math.max(0, entry.refCount - 1);
        return;
      }
    }
    // Buffer not in cache - it may have been created outside the cache
    // Don't destroy it here; caller is responsible
  }

  /**
   * Evict least recently used entry.
   * IMPORTANT: Buffer is NOT destroyed immediately - it's queued for deferred
   * destruction to avoid use-after-destroy bugs with pending command buffers.
   */
  #evictLRU() {
    /** @type {string | null} */
    let oldestHash = null;
    let oldestTime = Infinity;

    for (const [hash, entry] of this.#cache) {
      // Prefer evicting entries with refCount 0
      if (entry.refCount === 0 && entry.lastUsed < oldestTime) {
        oldestTime = entry.lastUsed;
        oldestHash = hash;
      }
    }

    // If all entries are in use, evict oldest anyway
    if (oldestHash === null) {
      for (const [hash, entry] of this.#cache) {
        if (entry.lastUsed < oldestTime) {
          oldestTime = entry.lastUsed;
          oldestHash = hash;
        }
      }
    }

    if (oldestHash) {
      const entry = this.#cache.get(oldestHash);
      if (entry) {
        // DON'T destroy immediately - defer until GPU work completes
        this.#pendingDestruction.push(entry.buffer);
        this.#cache.delete(oldestHash);
        this.#stats.evictions++;
        this.#stats.currentSize = this.#cache.size;
      }
    }
  }

  /**
   * Evict stale entries (older than maxAgeMs).
   * Buffers are queued for deferred destruction.
   * @returns {number}
   */
  evictStale() {
    const now = performance.now();
    let evicted = 0;

    for (const [hash, entry] of this.#cache) {
      if (entry.refCount === 0 && now - entry.lastUsed > this.#maxAgeMs) {
        // DON'T destroy immediately - defer until GPU work completes
        this.#pendingDestruction.push(entry.buffer);
        this.#cache.delete(hash);
        evicted++;
      }
    }

    this.#stats.evictions += evicted;
    this.#stats.currentSize = this.#cache.size;
    return evicted;
  }

  /**
   * Clear all cached buffers.
   * Also flushes any pending destruction queue.
   * @returns {void}
   */
  clear() {
    // Flush pending destruction first
    this.flushPendingDestruction();

    // Destroy all cached buffers
    for (const entry of this.#cache.values()) {
      entry.buffer.destroy();
    }
    this.#cache.clear();
    this.#stats.currentSize = 0;
  }

  /**
   * Destroy all buffers in the pending destruction queue.
   * Call this after GPU work completes (e.g., after onSubmittedWorkDone).
   *
   * This is critical for avoiding use-after-destroy bugs: when the uniform
   * cache evicts a buffer that's still referenced by a pending command buffer,
   * the buffer is queued here instead of being destroyed immediately.
   * @returns {number}
   */
  flushPendingDestruction() {
    const count = this.#pendingDestruction.length;
    for (const buffer of this.#pendingDestruction) {
      buffer.destroy();
    }
    this.#pendingDestruction = [];
    return count;
  }

  /**
   * Get the number of buffers pending destruction.
   * @returns {number}
   */
  getPendingDestructionCount() {
    return this.#pendingDestruction.length;
  }

  /**
   * Check if a buffer is managed by this cache
   * @param {GPUBuffer} buffer
   * @returns {boolean}
   */
  isCached(buffer) {
    for (const entry of this.#cache.values()) {
      if (entry.buffer === buffer) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get cache statistics
   * @returns {{hits: number, misses: number, evictions: number, currentSize: number, hitRate: string, pendingDestruction: number}}
   */
  getStats() {
    const total = this.#stats.hits + this.#stats.misses;
    const hitRate = total > 0 ? ((this.#stats.hits / total) * 100).toFixed(1) + '%' : '0%';
    return { ...this.#stats, hitRate, pendingDestruction: this.#pendingDestruction.length };
  }
}

/**
 * Release or destroy a uniform buffer appropriately.
 * If the buffer is cached, releases it back to the cache.
 * If not cached, destroys it directly.
 * @param {GPUBuffer} buffer
 * @returns {void}
 */
export function releaseUniformBuffer(buffer) {
  const cache = getUniformCache();
  if (cache.isCached(buffer)) {
    cache.release(buffer);
  } else {
    buffer.destroy();
  }
}

// Global singleton instance
/** @type {UniformBufferCache | null} */
let globalUniformCache = null;

/**
 * Get the global uniform buffer cache instance
 * @returns {UniformBufferCache}
 */
export function getUniformCache() {
  if (!globalUniformCache) {
    globalUniformCache = new UniformBufferCache();
  }
  return globalUniformCache;
}

/**
 * Reset the global uniform cache (useful for testing or device loss)
 * @returns {void}
 */
export function resetUniformCache() {
  if (globalUniformCache) {
    globalUniformCache.clear();
    globalUniformCache = null;
  }
}
