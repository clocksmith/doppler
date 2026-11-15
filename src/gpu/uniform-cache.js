

import { getDevice } from './device.js';
import { isDeviceLost, observeDeviceLoss, registerBufferDevice } from '../memory/device-residency.js';
import { getRuntimeConfig } from '../config/runtime.js';

const uniformOwners = new WeakMap();
const uniformContents = new WeakMap();

export function peekUniformCacheBytes(buffer) {
  return uniformContents.get(buffer) ?? null;
}


function hashUniformBytes(view) {
  let hash = 2166136261; // FNV offset basis

  for (let i = 0; i < view.length; i++) {
    hash ^= view[i];
    hash = Math.imul(hash, 16777619); // FNV prime
  }

  // Convert to hex string for Map key
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function copyUniformBytes(view) {
  return new Uint8Array(view);
}

export function toUniformArrayBuffer(data) {
  if (data instanceof ArrayBuffer) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    if (data.byteOffset === 0 && data.byteLength === data.buffer.byteLength) {
      return data.buffer;
    }
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  throw new Error('Uniform data must be ArrayBuffer or ArrayBufferView.');
}

function equalUniformBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}


export class UniformBufferCache {
  
  #cache = new Map();

  
  #stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    currentSize: 0,
    transientFallbacks: 0,
  };

  
  #pendingDestruction = [];

  
  #maxEntries;

  
  #maxAgeMs;
  #device;
  #completedLeases;
  #leaseEntries = new WeakMap();
  #transientLeases = new WeakSet();
  #outstandingLeases = 0;
  #retiredActiveEntries = 0;
  #transientOutstandingLeases = 0;

  
  constructor(
    maxEntries = getRuntimeConfig().shared.gpuCache.uniformCacheMaxEntries,
    maxAgeMs = getRuntimeConfig().shared.gpuCache.uniformCacheMaxAgeMs,
    device = getDevice(),
    completedLeases = false
  ) {
    this.#maxEntries = maxEntries;
    this.#maxAgeMs = maxAgeMs;
    this.#device = device;
    this.#completedLeases = completedLeases;
    observeDeviceLoss(device);
    device?.lost?.then(() => this.clear(), () => this.clear());
  }

  
  getOrCreate(data, label) {
    if (isDeviceLost(this.#device)) throw new Error('Uniform cache GPU device is lost.');
    const dataView = new Uint8Array(data);
    const baseKey = `${data.byteLength}:${hashUniformBytes(dataView)}`;
    let key = baseKey;
    let suffix = 0;
    let existing = this.#cache.get(key);

    while (existing) {
      if (equalUniformBytes(existing.bytes, dataView)) {
        existing.lastUsed = performance.now();
        existing.refCount++;
        this.#stats.hits++;
        if (this.#completedLeases) this.#outstandingLeases++;
        return existing.buffer;
      }
      suffix += 1;
      key = `${baseKey}#${suffix}`;
      existing = this.#cache.get(key);
    }

    const admitted = !this.#completedLeases || this.#cache.size < this.#maxEntries || this.#evictLRU();
    const dataBytes = copyUniformBytes(dataView);

    // Cache miss - create new buffer
    this.#stats.misses++;

    const device = this.#device;
    if (!device) {
      throw new Error('GPU device not initialized');
    }

    const buffer = device.createBuffer({
      label: `${label}_cached`,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    try {
      registerBufferDevice(buffer, device);
      device.queue.writeBuffer(buffer, 0, data);
    } catch (error) {
      buffer.destroy();
      throw error;
    }
    // If every cached lease is still in flight, preserve the bound and return
    // an uncached buffer. Owner-aware release destroys it after completion.
    if (!admitted) {
      this.#stats.transientFallbacks++;
      this.#outstandingLeases++;
      this.#transientOutstandingLeases++;
      this.#transientLeases.add(buffer);
      uniformOwners.set(buffer, this);
      return buffer;
    }
    uniformOwners.set(buffer, this);
    uniformContents.set(buffer, dataBytes);

    // Evict if at capacity
    if (!this.#completedLeases && this.#cache.size >= this.#maxEntries) {
      this.#evictLRU();
    }

    const entry = {
      buffer,
      bytes: dataBytes,
      lastUsed: performance.now(),
      refCount: 1,
      retired: false,
    };
    this.#cache.set(key, entry);
    if (this.#completedLeases) {
      this.#leaseEntries.set(buffer, entry);
      this.#outstandingLeases++;
    }
    this.#stats.currentSize = this.#cache.size;

    return buffer;
  }

  
  release(buffer) {
    if (this.#completedLeases) {
      if (this.#transientLeases.delete(buffer)) {
        this.#outstandingLeases--;
        this.#transientOutstandingLeases--;
        buffer.destroy();
        return;
      }
      const entry = this.#leaseEntries.get(buffer);
      if (!entry || entry.refCount === 0) return;
      entry.refCount--;
      this.#outstandingLeases--;
      if (entry.retired && entry.refCount === 0) {
        this.#retiredActiveEntries--;
        buffer.destroy();
      }
      return;
    }
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

    if (oldestHash === null && this.#completedLeases) return false;

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
        if (this.#completedLeases) { entry.retired = true; entry.buffer.destroy(); }
        else this.#pendingDestruction.push(entry.buffer);
        this.#cache.delete(oldestHash);
        this.#stats.evictions++;
        this.#stats.currentSize = this.#cache.size;
        return true;
      }
    }
    return false;
  }


  clear() {
    // Flush pending destruction first
    this.flushPendingDestruction();

    // Destroy all cached buffers
    for (const entry of this.#cache.values()) {
      if (this.#completedLeases) {
        entry.retired = true;
        if (entry.refCount > 0) this.#retiredActiveEntries++;
      }
      if (!this.#completedLeases || entry.refCount === 0) entry.buffer.destroy();
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
    const leaseStats = this.#completedLeases ? {
      outstandingLeases: this.#outstandingLeases,
      retiredActiveEntries: this.#retiredActiveEntries,
      transientOutstandingLeases: this.#transientOutstandingLeases,
    } : {};
    return { ...this.#stats, ...leaseStats, hitRate, pendingDestruction: this.#pendingDestruction.length };
  }
}


export function releaseUniformBuffer(buffer) {
  const cache = uniformOwners.get(buffer);
  if (cache) {
    cache.release(buffer);
  } else {
    buffer.destroy();
  }
}

// Global singleton instance

const deviceUniformCaches = new WeakMap();


export function getUniformCache(device = getDevice()) {
  if (!device || isDeviceLost(device)) throw new Error('Uniform cache requires a live GPU device.');
  let cache = deviceUniformCaches.get(device);
  if (!cache) {
    cache = new UniformBufferCache(undefined, undefined, device);
    deviceUniformCaches.set(device, cache);
  }
  return cache;
}

export function getUniformCacheStats(device = getDevice()) {
  return deviceUniformCaches.get(device)?.getStats() ?? null;
}


export function resetUniformCache(device = getDevice()) {
  deviceUniformCaches.get(device)?.clear();
  if (device) deviceUniformCaches.delete(device);
}

// Separate ownership domain: only wrappers releasing after queue completion use
// this cache. Legacy recorded callers and their flush lifecycle remain separate.
const completedUniformCaches = new WeakMap();
export function getCompletedUniformCache(device) {
  if (!device || isDeviceLost(device)) throw new Error('Completed uniform cache requires a live device.');
  let cache = completedUniformCaches.get(device);
  if (!cache) {
    cache = new UniformBufferCache(undefined, undefined, device, true);
    completedUniformCaches.set(device, cache);
  }
  return cache;
}

export function getCompletedUniformCacheStats(device) {
  return completedUniformCaches.get(device)?.getStats() ?? null;
}
