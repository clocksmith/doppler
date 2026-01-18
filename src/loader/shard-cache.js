import {
  loadShard as loadShardFromStore,
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

  lastSource = null;

  constructor(config) {
    this.#maxEntries = config.maxEntries;
    this.#customLoader = config.customLoader ?? null;
    this.#verifyHashes = config.verifyHashes
      ?? config.loadingConfig?.verifyHashes
      ?? true;
    this.#manifest = config.manifest ?? null;
    this.#loadingConfig = config.loadingConfig ?? getRuntimeConfig().loading.shardCache;
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

  async load(shardIndex) {
    const shardInfo = this.#manifest?.shards?.[shardIndex];
    const sizeStr = shardInfo ? formatBytes(shardInfo.size) : '';

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
    const fetchPromise = this.#doLoad(shardIndex, sizeStr);
    this.#fetchPromises.set(shardIndex, fetchPromise);

    try {
      const result = await fetchPromise;
      return result;
    } finally {
      // Remove from in-flight map when done (success or error)
      this.#fetchPromises.delete(shardIndex);
    }
  }

  async #doLoad(shardIndex, sizeStr) {
    if (this.#customLoader) {
      const startTime = performance.now();
      let data = await this.#customLoader(shardIndex);

      // Verify hash if enabled
      if (this.#verifyHashes && this.#manifest) {
        const shardInfo = this.#manifest.shards?.[shardIndex];
        const expectedHash = shardInfo?.hash || shardInfo?.blake3;
        if (expectedHash) {
          const algorithm = this.#manifest.hashAlgorithm || 'blake3';
          const computedHash = await computeHash(data, algorithm);
          if (computedHash !== expectedHash) {
            throw new Error(
              `Shard ${shardIndex} hash mismatch. Expected: ${expectedHash}, got: ${computedHash}`
            );
          }
        }
      }

      // Normalize to ArrayBuffer for downstream slicing
      let arrayBuffer;
      if (data instanceof Uint8Array) {
        arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
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
  });
}
