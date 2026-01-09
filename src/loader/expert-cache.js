/**
 * Expert LRU Cache for MoE Models
 *
 * Tracks expert residency in VRAM and implements LRU eviction
 * to manage memory pressure during inference.
 *
 * @module loader/expert-cache
 */

import { releaseBuffer } from '../gpu/buffer-pool.js';
import { log, trace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

/**
 * @typedef {Object} CacheEntry
 * @property {import('./weights.js').ExpertWeights} weights
 * @property {number} lastAccess
 * @property {number} sizeBytes
 */

/**
 * Expert LRU Cache
 *
 * Manages expert weight residency in VRAM with LRU eviction policy.
 */
export class ExpertCache {
  /** @type {Map<string, CacheEntry>} */
  #cache = new Map();

  /** @type {number} */
  #maxBytes;

  /** @type {number} */
  #currentBytes = 0;

  /** @type {number} */
  #accessCounter = 0;

  /** @type {import('../config/schema/loading.schema.js').ExpertCacheConfigSchema} */
  #config;

  // Statistics
  /** @type {number} */
  #hits = 0;

  /** @type {number} */
  #misses = 0;

  /** @type {number} */
  #evictions = 0;

  /** @type {Set<string>} */
  #inUse = new Set();

  /** @type {Set<string>} */
  #pinned = new Set();

  /**
   * Create expert cache
   * @param {number} [maxBytes] Maximum cache size in bytes (uses config default if not specified)
   * @param {import('../config/schema/loading.schema.js').ExpertCacheConfigSchema} [config] Expert cache configuration
   */
  constructor(maxBytes, config) {
    this.#config = config ?? getRuntimeConfig().loading.expertCache;
    this.#maxBytes = maxBytes ?? this.#config.defaultSizeBytes;
  }

  /**
   * Update cache configuration at runtime.
   * @param {import('../config/schema/loading.schema.js').ExpertCacheConfigSchema} config
   * @param {number} [maxBytes]
   */
  configure(config, maxBytes) {
    this.#config = config;
    this.#maxBytes = maxBytes ?? config.defaultSizeBytes;
  }

  /**
   * Auto-tune cache size based on available VRAM
   * Call this after WebGPU is initialized
   */
  async autoTune() {
    const { defaultSizeBytes, maxBufferPercentage } = this.#config;
    const defaultSizeMB = (defaultSizeBytes / 1024 / 1024).toFixed(0);

    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      log.info('ExpertCache', `WebGPU not available, using default ${defaultSizeMB}MB`);
      return;
    }

    try {
      const adapter = await /** @type {any} */ (navigator).gpu.requestAdapter();
      if (!adapter) {
        log.info('ExpertCache', `No GPU adapter, using default ${defaultSizeMB}MB`);
        return;
      }

      const limits = adapter.limits;
      const maxBufferSize = limits?.maxBufferSize || 256 * 1024 * 1024;

      // Heuristic: Use up to default size or configured percentage of max buffer size, whichever is smaller
      // This leaves room for model weights, KV cache, and activations
      const autoSize = Math.min(
        defaultSizeBytes,
        Math.floor(maxBufferSize * maxBufferPercentage)
      );

      this.#maxBytes = autoSize;
      log.info('ExpertCache', `Auto-tuned to ${(this.#maxBytes / 1024 / 1024).toFixed(0)}MB (maxBuffer: ${(maxBufferSize / 1024 / 1024).toFixed(0)}MB)`);
    } catch (e) {
      log.warn('ExpertCache', `Auto-tune failed, using default ${defaultSizeMB}MB:`, e);
    }
  }

  /**
   * Generate cache key for expert
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @returns {string}
   */
  #getKey(layerIdx, expertIdx) {
    return `${layerIdx}_${expertIdx}`;
  }

  /**
   * Get expert from cache
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @returns {import('./weights.js').ExpertWeights | null} Expert weights or null if not in cache
   */
  get(layerIdx, expertIdx) {
    const key = this.#getKey(layerIdx, expertIdx);
    const entry = this.#cache.get(key);

    if (entry) {
      // Update access time for LRU tracking
      entry.lastAccess = ++this.#accessCounter;
      this.#hits++;
      return entry.weights;
    }

    this.#misses++;
    return null;
  }

  /**
   * Put expert into cache
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @param {import('./weights.js').ExpertWeights} weights Expert weights to cache
   * @param {number} sizeBytes Size of expert in bytes (for memory tracking)
   */
  put(layerIdx, expertIdx, weights, sizeBytes) {
    const key = this.#getKey(layerIdx, expertIdx);
    const existing = this.#cache.get(key);
    let existingSize = existing?.sizeBytes ?? 0;

    // If already in cache, update it
    let projectedBytes = this.#currentBytes - existingSize + sizeBytes;
    while (projectedBytes > this.#maxBytes && this.#cache.size > 0) {
      const evicted = this.evictLRU();
      if (!evicted) {
        log.warn('ExpertCache', `Cache full; unable to evict for ${key}. Skipping cache insert.`);
        return;
      }
      if (!this.#cache.has(key)) {
        existingSize = 0;
      }
      projectedBytes = this.#currentBytes - existingSize + sizeBytes;
    }

    // Add to cache
    this.#cache.set(key, {
      weights,
      lastAccess: ++this.#accessCounter,
      sizeBytes,
    });
    this.#currentBytes = this.#currentBytes - existingSize + sizeBytes;
  }

  /**
   * Check if expert is in cache
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @returns {boolean}
   */
  has(layerIdx, expertIdx) {
    return this.#cache.has(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Evict least recently used expert
   * Skips experts that are in-use or pinned
   * @returns {boolean} true if an expert was evicted, false if all experts are protected
   */
  evictLRU() {
    if (this.#cache.size === 0) return false;

    /** @type {string | null} */
    let lruKey = null;
    let lruTime = Infinity;

    for (const [key, entry] of this.#cache) {
      // Skip in-use experts (currently being used in inference)
      if (this.#inUse.has(key)) continue;
      // Skip pinned experts (shared experts that should never be evicted)
      if (this.#pinned.has(key)) continue;

      if (entry.lastAccess < lruTime) {
        lruTime = entry.lastAccess;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.#evict(lruKey);
      return true;
    }

    // All experts are either in-use or pinned
    return false;
  }

  /**
   * Mark expert as in-use (prevents eviction during inference)
   * @param {number} layerIdx
   * @param {number} expertIdx
   */
  markInUse(layerIdx, expertIdx) {
    this.#inUse.add(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Mark expert as no longer in use (allows eviction)
   * @param {number} layerIdx
   * @param {number} expertIdx
   */
  markNotInUse(layerIdx, expertIdx) {
    this.#inUse.delete(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Clear all in-use markers (call after inference completes)
   */
  clearInUse() {
    this.#inUse.clear();
  }

  /**
   * Pin expert (prevents eviction, for shared experts)
   * @param {number} layerIdx
   * @param {number} expertIdx
   */
  pinExpert(layerIdx, expertIdx) {
    this.#pinned.add(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Unpin expert (allows eviction)
   * @param {number} layerIdx
   * @param {number} expertIdx
   */
  unpinExpert(layerIdx, expertIdx) {
    this.#pinned.delete(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Pin all shared experts for a model
   * @param {number[]} sharedExpertIndices
   * @param {number} numLayers
   */
  pinSharedExperts(sharedExpertIndices, numLayers) {
    for (let layer = 0; layer < numLayers; layer++) {
      for (const expertIdx of sharedExpertIndices) {
        this.pinExpert(layer, expertIdx);
      }
    }
    log.info('ExpertCache', `Pinned ${sharedExpertIndices.length} shared experts across ${numLayers} layers`);
  }

  /**
   * Check if expert is pinned
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @returns {boolean}
   */
  isPinned(layerIdx, expertIdx) {
    return this.#pinned.has(this.#getKey(layerIdx, expertIdx));
  }

  /**
   * Evict specific expert by key
   * @param {string} key
   */
  #evict(key) {
    const entry = this.#cache.get(key);
    if (!entry) return;

    // Release GPU buffers
    this.#releaseExpertBuffers(entry.weights);

    this.#currentBytes -= entry.sizeBytes;
    this.#cache.delete(key);
    this.#evictions++;

    trace.loader(`Evicted expert ${key}, freed ${(entry.sizeBytes / 1024 / 1024).toFixed(1)}MB`);
  }

  /**
   * Release GPU buffers for expert weights
   * @param {import('./weights.js').ExpertWeights} weights
   */
  #releaseExpertBuffers(weights) {
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
   * @returns {number}
   */
  getMemoryUsage() {
    return this.#currentBytes;
  }

  /**
   * Get cache statistics
   * @returns {import('./expert-cache.js').CacheStats}
   */
  getStats() {
    const total = this.#hits + this.#misses;
    return {
      hits: this.#hits,
      misses: this.#misses,
      evictions: this.#evictions,
      currentSize: this.#currentBytes,
      maxSize: this.#maxBytes,
      expertCount: this.#cache.size,
      hitRate: total > 0 ? this.#hits / total : 0,
      inUseCount: this.#inUse.size,
      pinnedCount: this.#pinned.size,
    };
  }

  /**
   * Clear all cached experts
   */
  clear() {
    for (const [, entry] of this.#cache) {
      this.#releaseExpertBuffers(entry.weights);
    }
    this.#cache.clear();
    this.#currentBytes = 0;
    this.#inUse.clear();
    // Note: pinned is NOT cleared - shared experts stay pinned
    log.info('ExpertCache', 'Cache cleared');
  }

  /**
   * Set maximum cache size
   * @param {number} maxBytes New maximum size in bytes
   */
  setMaxSize(maxBytes) {
    this.#maxBytes = maxBytes;

    // Evict if over new limit
    while (this.#currentBytes > this.#maxBytes && this.#cache.size > 0) {
      this.evictLRU();
    }
  }

  /**
   * Prefetch experts (hint for future access)
   * This is a no-op in the cache - actual prefetch happens in the loader
   * @param {number} _layerIdx
   * @param {number[]} _expertIndices
   */
  prefetch(_layerIdx, _expertIndices) {
    // Prefetch hint - the loader should implement actual prefetch logic
  }

  /**
   * Get all cached expert keys
   * @returns {Array<{ layerIdx: number; expertIdx: number }>}
   */
  getCachedExperts() {
    /** @type {Array<{ layerIdx: number; expertIdx: number }>} */
    const result = [];
    for (const key of this.#cache.keys()) {
      const [layer, expert] = key.split('_').map(Number);
      result.push({ layerIdx: layer, expertIdx: expert });
    }
    return result;
  }
}

/** @type {ExpertCache | null} */
let globalCache = null;

/**
 * Get global expert cache instance
 * @param {import('../config/schema/loading.schema.js').ExpertCacheConfigSchema} [config]
 * @returns {ExpertCache}
 */
export function getExpertCache(config) {
  if (!globalCache) {
    const resolvedConfig = config ?? getRuntimeConfig().loading.expertCache;
    globalCache = new ExpertCache(undefined, resolvedConfig);
  } else {
    const resolvedConfig = config ?? getRuntimeConfig().loading.expertCache;
    globalCache.configure(resolvedConfig);
  }
  return globalCache;
}

/**
 * Create new expert cache with custom size
 * @param {number} [maxBytes]
 * @param {import('../config/schema/loading.schema.js').ExpertCacheConfigSchema} [config]
 * @returns {ExpertCache}
 */
export function createExpertCache(maxBytes, config) {
  const resolvedConfig = config ?? getRuntimeConfig().loading.expertCache;
  return new ExpertCache(maxBytes, resolvedConfig);
}

export default ExpertCache;
