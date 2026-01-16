

import { getDevice } from './device.js';
import { getRuntimeConfig } from '../config/runtime.js';


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


export class UniformBufferCache {
  
  #cache = new Map();

  
  #stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    currentSize: 0,
  };

  
  #pendingDestruction = [];

  
  #maxEntries;

  
  #maxAgeMs;

  
  constructor(
    maxEntries = getRuntimeConfig().shared.gpuCache.uniformCacheMaxEntries,
    maxAgeMs = getRuntimeConfig().shared.gpuCache.uniformCacheMaxAgeMs
  ) {
    this.#maxEntries = maxEntries;
    this.#maxAgeMs = maxAgeMs;
  }

  
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

  
  #evictLRU() {
    
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

  
  flushPendingDestruction() {
    const count = this.#pendingDestruction.length;
    for (const buffer of this.#pendingDestruction) {
      buffer.destroy();
    }
    this.#pendingDestruction = [];
    return count;
  }

  
  getPendingDestructionCount() {
    return this.#pendingDestruction.length;
  }

  
  isCached(buffer) {
    for (const entry of this.#cache.values()) {
      if (entry.buffer === buffer) {
        return true;
      }
    }
    return false;
  }

  
  getStats() {
    const total = this.#stats.hits + this.#stats.misses;
    const hitRate = total > 0 ? ((this.#stats.hits / total) * 100).toFixed(1) + '%' : '0%';
    return { ...this.#stats, hitRate, pendingDestruction: this.#pendingDestruction.length };
  }
}


export function releaseUniformBuffer(buffer) {
  const cache = getUniformCache();
  if (cache.isCached(buffer)) {
    cache.release(buffer);
  } else {
    buffer.destroy();
  }
}

// Global singleton instance

let globalUniformCache = null;


export function getUniformCache() {
  if (!globalUniformCache) {
    globalUniformCache = new UniformBufferCache();
  }
  return globalUniformCache;
}


export function resetUniformCache() {
  if (globalUniformCache) {
    globalUniformCache.clear();
    globalUniformCache = null;
  }
}
