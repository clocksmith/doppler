

import { releaseBuffer } from '../gpu/buffer-pool.js';
import { log, trace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';




export class ExpertCache {
  
  #cache = new Map();

  
  #maxBytes;

  
  #currentBytes = 0;

  
  #accessCounter = 0;

  
  #config;

  // Statistics
  
  #hits = 0;

  
  #misses = 0;

  
  #evictions = 0;

  
  #inUse = new Set();

  
  #pinned = new Set();

  
  constructor(maxBytes, config) {
    this.#config = config ?? getRuntimeConfig().loading.expertCache;
    this.#maxBytes = maxBytes ?? this.#config.defaultSizeBytes;
  }

  
  configure(config, maxBytes) {
    this.#config = config;
    this.#maxBytes = maxBytes ?? config.defaultSizeBytes;
  }

  
  async autoTune() {
    const { defaultSizeBytes, maxBufferPercentage } = this.#config;
    const defaultSizeMB = (defaultSizeBytes / 1024 / 1024).toFixed(0);

    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      log.info('ExpertCache', `WebGPU not available, using default ${defaultSizeMB}MB`);
      return;
    }

    try {
      const adapter = await  (navigator).gpu.requestAdapter();
      if (!adapter) {
        log.info('ExpertCache', `No GPU adapter, using default ${defaultSizeMB}MB`);
        return;
      }

      const limits = adapter.limits;
      const maxBufferSize = limits?.maxBufferSize ?? this.#config.maxBufferFallbackBytes;

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

  
  #getKey(layerIdx, expertIdx) {
    return `${layerIdx}_${expertIdx}`;
  }

  
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

  
  has(layerIdx, expertIdx) {
    return this.#cache.has(this.#getKey(layerIdx, expertIdx));
  }

  
  evictLRU() {
    if (this.#cache.size === 0) return false;

    
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

  
  markInUse(layerIdx, expertIdx) {
    this.#inUse.add(this.#getKey(layerIdx, expertIdx));
  }

  
  markNotInUse(layerIdx, expertIdx) {
    this.#inUse.delete(this.#getKey(layerIdx, expertIdx));
  }

  
  clearInUse() {
    this.#inUse.clear();
  }

  
  pinExpert(layerIdx, expertIdx) {
    this.#pinned.add(this.#getKey(layerIdx, expertIdx));
  }

  
  unpinExpert(layerIdx, expertIdx) {
    this.#pinned.delete(this.#getKey(layerIdx, expertIdx));
  }

  
  pinSharedExperts(sharedExpertIndices, numLayers) {
    for (let layer = 0; layer < numLayers; layer++) {
      for (const expertIdx of sharedExpertIndices) {
        this.pinExpert(layer, expertIdx);
      }
    }
    log.info('ExpertCache', `Pinned ${sharedExpertIndices.length} shared experts across ${numLayers} layers`);
  }

  
  isPinned(layerIdx, expertIdx) {
    return this.#pinned.has(this.#getKey(layerIdx, expertIdx));
  }

  
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

  
  getMemoryUsage() {
    return this.#currentBytes;
  }

  
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

  
  setMaxSize(maxBytes) {
    this.#maxBytes = maxBytes;

    // Evict if over new limit
    while (this.#currentBytes > this.#maxBytes && this.#cache.size > 0) {
      this.evictLRU();
    }
  }

  
  prefetch(_layerIdx, _expertIndices) {
    // Prefetch hint - the loader should implement actual prefetch logic
  }

  
  getCachedExperts() {
    
    const result = [];
    for (const key of this.#cache.keys()) {
      const [layer, expert] = key.split('_').map(Number);
      result.push({ layerIdx: layer, expertIdx: expert });
    }
    return result;
  }
}


let globalCache = null;


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


export function createExpertCache(maxBytes, config) {
  const resolvedConfig = config ?? getRuntimeConfig().loading.expertCache;
  return new ExpertCache(maxBytes, resolvedConfig);
}

export default ExpertCache;
