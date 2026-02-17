import {
  loadShard as loadShardFromStore,
  loadShardRange as loadShardRangeFromStore,
  computeHash,
  getStorageBackendType,
} from '../storage/shard-manager.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

export class ShardCache {
  #cache = new Map();
  #maxEntries;
  #customLoader = null;
  #verifyHashes;
  #manifest = null;
  #loadingConfig;
  #fetchPromises = new Map();
  #maxConcurrentLoads = 0;
  #inFlightLoads = 0;
  #highPriorityQueue = [];
  #lowPriorityQueue = [];

  lastSource = null;

  constructor(config) {
    this.#maxEntries = config.maxEntries;
    this.#customLoader = config.customLoader ?? null;
    this.#verifyHashes = config.verifyHashes
      ?? config.loadingConfig?.verifyHashes
      ?? true;
    this.#manifest = config.manifest ?? null;
    this.#loadingConfig = config.loadingConfig ?? getRuntimeConfig().loading.shardCache;
    this.#maxConcurrentLoads = config.maxConcurrentLoads
      ?? config.loadingConfig?.maxConcurrentLoads
      ?? 0;
  }

  configure(config) {
    if (config.maxEntries !== undefined) {
      this.#maxEntries = config.maxEntries;
    }
    if (config.customLoader !== undefined) {
      this.#customLoader = config.customLoader;
    }
    if (config.verifyHashes !== undefined) {
      this.#verifyHashes = config.verifyHashes;
    }
    if (config.manifest !== undefined) {
      this.#manifest = config.manifest;
    }
    if (config.loadingConfig !== undefined) {
      this.#loadingConfig = config.loadingConfig;
      if (config.loadingConfig.maxConcurrentLoads !== undefined) {
        this.#maxConcurrentLoads = config.loadingConfig.maxConcurrentLoads;
      }
      this.#drainQueue();
    }
    if (config.maxConcurrentLoads !== undefined) {
      this.#maxConcurrentLoads = config.maxConcurrentLoads;
      this.#drainQueue();
    }
  }

  setCustomLoader(loader, verify = true) {
    this.#customLoader = loader;
    this.#verifyHashes = verify;
    if (loader) {
      log.info('ShardCache', 'Custom shard loader configured');
    }
  }

  setManifest(manifest) {
    this.#manifest = manifest;
  }

  get hasCustomLoader() {
    return this.#customLoader !== null;
  }

  has(shardIndex) {
    return this.#cache.has(shardIndex);
  }

  get size() {
    return this.#cache.size;
  }

  get totalBytes() {
    return Array.from(this.#cache.values()).reduce((sum, ab) => sum + ab.byteLength, 0);
  }

  async load(shardIndex, options = {}) {
    const shardInfo = this.#manifest?.shards?.[shardIndex];
    const sizeStr = shardInfo ? formatBytes(shardInfo.size) : '';
    const priority = options.priority === 'low' ? 'low' : 'high';

    // 1. Check cache first
    if (this.#cache.has(shardIndex)) {
      const cached = this.#cache.get(shardIndex);
      // Refresh LRU order
      this.#cache.delete(shardIndex);
      this.#cache.set(shardIndex, cached);
      this.lastSource = { source: 'RAM', elapsed: 0 };
      log.verbose('ShardCache', `Shard ${shardIndex}: RAM${sizeStr ? ` (${sizeStr})` : ''}`);
      return cached;
    }

    // 2. Check if fetch is already in-flight - deduplicate concurrent requests
    if (this.#fetchPromises.has(shardIndex)) {
      log.verbose('ShardCache', `Shard ${shardIndex}: waiting for in-flight fetch`);
      return this.#fetchPromises.get(shardIndex);
    }

    // 3. Start the actual fetch and store the promise for deduplication
    const fetchPromise = this.#scheduleLoad(
      priority,
      () => this.#doLoad(shardIndex, sizeStr)
    );
    this.#fetchPromises.set(shardIndex, fetchPromise);

    try {
      const result = await fetchPromise;
      return result;
    } finally {
      // Remove from in-flight map when done (success or error)
      this.#fetchPromises.delete(shardIndex);
    }
  }

  /**
   * Load a byte range from a shard.
   *
   * If the full shard is already cached, this returns a slice without touching storage.
   * Otherwise, this will attempt a backend range read (when available) to avoid
   * materializing the full shard in RAM.
   */
  async loadRange(shardIndex, offset = 0, length = null, options = {}) {
    const start = Math.max(0, offset | 0);
    const want = length == null ? null : Math.max(0, length | 0);

    if (this.#cache.has(shardIndex)) {
      const cached = this.#cache.get(shardIndex);
      // Refresh LRU order
      this.#cache.delete(shardIndex);
      this.#cache.set(shardIndex, cached);
      this.lastSource = { source: 'RAM', elapsed: 0 };
      const view = new Uint8Array(cached);
      const end = want == null ? view.length : Math.min(view.length, start + want);
      // Return a compact ArrayBuffer (downstream expects independent buffers).
      return view.slice(start, end).buffer;
    }

    if (this.#customLoader) {
      // Custom loaders only support whole-shard loads; fall back to full shard then slice.
      const full = await this.load(shardIndex, options);
      const view = new Uint8Array(full);
      const end = want == null ? view.length : Math.min(view.length, start + want);
      return view.slice(start, end).buffer;
    }

    // Direct backend range read (no shard cache population).
    return loadShardRangeFromStore(shardIndex, start, want, options);
  }

  prefetch(shardIndex) {
    return this.load(shardIndex, { priority: 'low' });
  }

  async #doLoad(shardIndex, sizeStr) {
    if (this.#customLoader) {
      const startTime = performance.now();
      let data = await this.#customLoader(shardIndex);

      // Verify hash if enabled
      if (this.#verifyHashes && this.#manifest) {
        const shardInfo = this.#manifest.shards?.[shardIndex];
        const expectedHash = shardInfo?.hash;
        if (!expectedHash) {
          throw new Error(`Shard ${shardIndex} missing hash in manifest.`);
        }
        const algorithm = shardInfo?.hashAlgorithm ?? this.#manifest.hashAlgorithm;
        if (!algorithm) {
          throw new Error(`Manifest missing hashAlgorithm for shard ${shardIndex}.`);
        }
        const computedHash = await computeHash(data, algorithm);
        if (computedHash !== expectedHash) {
          throw new Error(
            `Shard ${shardIndex} hash mismatch. Expected: ${expectedHash}, got: ${computedHash}`
          );
        }
      }

      // Normalize to ArrayBuffer for downstream slicing
      let arrayBuffer;
      if (data instanceof Uint8Array) {
        // Avoid copying when the Uint8Array already covers the full buffer
        arrayBuffer = (data.byteOffset === 0 && data.byteLength === data.buffer.byteLength)
          ? data.buffer
          : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      } else {
        arrayBuffer = data;
      }

      this.#add(shardIndex, arrayBuffer);

      const elapsed = (performance.now() - startTime) / 1000;
      this.lastSource = { source: 'custom', elapsed };
      log.verbose('ShardCache', `Shard ${shardIndex}: network (${sizeStr}, ${elapsed.toFixed(2)}s)`);
      return arrayBuffer;
    }

    const storageStart = performance.now();
    const data = await loadShardFromStore(shardIndex);
    this.#add(shardIndex, data);
    const elapsed = (performance.now() - storageStart) / 1000;
    const backend = getStorageBackendType() ?? 'storage';
    this.lastSource = { source: backend, elapsed };
    log.verbose('ShardCache', `Shard ${shardIndex}: ${backend} (${sizeStr}, ${elapsed.toFixed(2)}s)`);
    return data;
  }

  async #scheduleLoad(priority, task) {
    const limit = this.#maxConcurrentLoads > 0
      ? this.#maxConcurrentLoads
      : Number.POSITIVE_INFINITY;

    if (this.#inFlightLoads < limit) {
      this.#inFlightLoads++;
      try {
        return await task();
      } finally {
        this.#inFlightLoads--;
        this.#drainQueue();
      }
    }

    return new Promise((resolve, reject) => {
      const entry = { task, resolve, reject };
      if (priority === 'low') {
        this.#lowPriorityQueue.push(entry);
      } else {
        this.#highPriorityQueue.push(entry);
      }
    });
  }

  #drainQueue() {
    const limit = this.#maxConcurrentLoads > 0
      ? this.#maxConcurrentLoads
      : Number.POSITIVE_INFINITY;

    while (this.#inFlightLoads < limit) {
      const entry = this.#highPriorityQueue.shift() ?? this.#lowPriorityQueue.shift();
      if (!entry) return;

      this.#inFlightLoads++;
      Promise.resolve()
        .then(entry.task)
        .then(entry.resolve, entry.reject)
        .finally(() => {
          this.#inFlightLoads--;
          this.#drainQueue();
        });
    }
  }

  #add(shardIndex, data) {
    this.#cache.set(shardIndex, data);
    if (this.#cache.size > this.#maxEntries) {
      const oldestKey = this.#cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.#cache.delete(oldestKey);
      }
    }
  }

  clear() {
    const count = this.#cache.size;
    const bytes = this.totalBytes;
    this.#cache.clear();
    debugTrace.loader(`Cleared shard cache: ${count} shards, ${formatBytes(bytes)} freed`);
  }

  configureForModel(manifest, hasCustomLoader) {
    if (!manifest) return;
    this.#manifest = manifest;

    const { opfsEntries, networkEntries, moeMaxEntries } = this.#loadingConfig;

    const moe = manifest.moeConfig;
    if (moe && moe.numExpertsPerToken > 0) {
      // For MoE: cache 2x top-k experts (for current + next layer prefetch) + 1 dense shard
      const expertCacheSize = (moe.numExpertsPerToken * 2) + 1;
      // Cap at configurable maximum
      this.#maxEntries = Math.min(moeMaxEntries, Math.max(4, expertCacheSize));
      debugTrace.loader(`MoE shard cache: ${this.#maxEntries} entries (${moe.numExpertsPerToken} experts/token)`);
    } else if (hasCustomLoader) {
      // Network loading: use larger cache to avoid re-fetching shards.
      this.#maxEntries = networkEntries;
      debugTrace.loader(`Network shard cache: ${this.#maxEntries} entries (avoiding re-fetch)`);
    } else {
      // OPFS (disk) loading - keep small cache, disk reads are fast
      this.#maxEntries = opfsEntries;
    }
  }
}

export function createShardCache(maxEntries, loadingConfig) {
  const config = loadingConfig ?? getRuntimeConfig().loading.shardCache;
  return new ShardCache({
    maxEntries: maxEntries ?? config.opfsEntries,
    loadingConfig: config,
    verifyHashes: config.verifyHashes,
    maxConcurrentLoads: config.maxConcurrentLoads,
  });
}
