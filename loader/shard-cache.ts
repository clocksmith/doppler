/**
 * Shard Cache
 *
 * LRU cache for model shards with deduplication of in-flight requests.
 *
 * @module loader/shard-cache
 */

import {
  loadShard as loadShardFromOPFS,
  computeHash,
} from '../storage/shard-manager.js';
import type { RDRRManifest } from '../storage/rdrr-format.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import type { CustomShardLoader, ShardSourceInfo } from './loader-types.js';

/**
 * Configuration for shard cache
 */
export interface ShardCacheConfig {
  maxEntries: number;
  customLoader?: CustomShardLoader;
  verifyHashes?: boolean;
  manifest?: RDRRManifest;
}

/**
 * LRU cache for model shards with request deduplication
 */
export class ShardCache {
  private cache = new Map<number, ArrayBuffer>();
  private maxEntries: number;
  private customLoader: CustomShardLoader | null = null;
  private verifyHashes: boolean;
  private manifest: RDRRManifest | null = null;

  // In-flight shard fetches - deduplicates concurrent requests
  private fetchPromises = new Map<number, Promise<ArrayBuffer>>();

  // Track last shard source for progress reporting
  lastSource: ShardSourceInfo | null = null;

  constructor(config: ShardCacheConfig) {
    this.maxEntries = config.maxEntries;
    this.customLoader = config.customLoader ?? null;
    this.verifyHashes = config.verifyHashes ?? true;
    this.manifest = config.manifest ?? null;
  }

  /**
   * Update configuration
   */
  configure(config: Partial<ShardCacheConfig>): void {
    if (config.maxEntries !== undefined) {
      this.maxEntries = config.maxEntries;
    }
    if (config.customLoader !== undefined) {
      this.customLoader = config.customLoader;
    }
    if (config.verifyHashes !== undefined) {
      this.verifyHashes = config.verifyHashes;
    }
    if (config.manifest !== undefined) {
      this.manifest = config.manifest;
    }
  }

  /**
   * Set custom shard loader
   */
  setCustomLoader(loader: CustomShardLoader | null, verify = true): void {
    this.customLoader = loader;
    this.verifyHashes = verify;
    if (loader) {
      log.info('ShardCache', 'Custom shard loader configured');
    }
  }

  /**
   * Set manifest for hash verification
   */
  setManifest(manifest: RDRRManifest | null): void {
    this.manifest = manifest;
  }

  /**
   * Check if a custom loader is configured
   */
  get hasCustomLoader(): boolean {
    return this.customLoader !== null;
  }

  /**
   * Check if shard is cached
   */
  has(shardIndex: number): boolean {
    return this.cache.has(shardIndex);
  }

  /**
   * Get cache size
   */
  get size(): number {
    return this.cache.size;
  }

  /**
   * Get total cached bytes
   */
  get totalBytes(): number {
    return Array.from(this.cache.values()).reduce((sum, ab) => sum + ab.byteLength, 0);
  }

  /**
   * Load shard with caching and request deduplication.
   * If the same shard is requested concurrently, all callers wait for the same fetch.
   */
  async load(shardIndex: number): Promise<ArrayBuffer> {
    const shardInfo = this.manifest?.shards?.[shardIndex];
    const sizeStr = shardInfo ? formatBytes(shardInfo.size) : '';

    // 1. Check cache first
    if (this.cache.has(shardIndex)) {
      const cached = this.cache.get(shardIndex)!;
      // Refresh LRU order
      this.cache.delete(shardIndex);
      this.cache.set(shardIndex, cached);
      this.lastSource = { source: 'RAM', elapsed: 0 };
      log.verbose('ShardCache', `Shard ${shardIndex}: RAM${sizeStr ? ` (${sizeStr})` : ''}`);
      return cached;
    }

    // 2. Check if fetch is already in-flight - deduplicate concurrent requests
    if (this.fetchPromises.has(shardIndex)) {
      log.verbose('ShardCache', `Shard ${shardIndex}: waiting for in-flight fetch`);
      return this.fetchPromises.get(shardIndex)!;
    }

    // 3. Start the actual fetch and store the promise for deduplication
    const fetchPromise = this.doLoad(shardIndex, sizeStr);
    this.fetchPromises.set(shardIndex, fetchPromise);

    try {
      const result = await fetchPromise;
      return result;
    } finally {
      // Remove from in-flight map when done (success or error)
      this.fetchPromises.delete(shardIndex);
    }
  }

  /**
   * Actually load the shard (called after deduplication check)
   */
  private async doLoad(shardIndex: number, sizeStr: string): Promise<ArrayBuffer> {
    if (this.customLoader) {
      const startTime = performance.now();
      let data: Uint8Array | ArrayBuffer = await this.customLoader(shardIndex);

      // Verify hash if enabled
      if (this.verifyHashes && this.manifest) {
        const shardInfo = this.manifest.shards?.[shardIndex];
        const expectedHash = shardInfo?.hash || (shardInfo as { blake3?: string })?.blake3;
        if (expectedHash) {
          const algorithm = this.manifest.hashAlgorithm || 'blake3';
          const computedHash = await computeHash(data, algorithm);
          if (computedHash !== expectedHash) {
            throw new Error(
              `Shard ${shardIndex} hash mismatch. Expected: ${expectedHash}, got: ${computedHash}`
            );
          }
        }
      }

      // Normalize to ArrayBuffer for downstream slicing
      let arrayBuffer: ArrayBuffer;
      if (data instanceof Uint8Array) {
        arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
      } else {
        arrayBuffer = data;
      }

      this.add(shardIndex, arrayBuffer);

      const elapsed = (performance.now() - startTime) / 1000;
      this.lastSource = { source: 'custom', elapsed };
      log.verbose('ShardCache', `Shard ${shardIndex}: network (${sizeStr}, ${elapsed.toFixed(2)}s)`);
      return arrayBuffer;
    }

    const opfsStart = performance.now();
    const data = await loadShardFromOPFS(shardIndex);
    this.add(shardIndex, data);
    const elapsed = (performance.now() - opfsStart) / 1000;
    this.lastSource = { source: 'OPFS', elapsed };
    log.verbose('ShardCache', `Shard ${shardIndex}: OPFS (${sizeStr}, ${elapsed.toFixed(2)}s)`);
    return data;
  }

  /**
   * Add shard to cache with LRU eviction
   */
  private add(shardIndex: number, data: ArrayBuffer): void {
    this.cache.set(shardIndex, data);
    if (this.cache.size > this.maxEntries) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.delete(oldestKey);
      }
    }
  }

  /**
   * Clear the cache
   */
  clear(): void {
    const count = this.cache.size;
    const bytes = this.totalBytes;
    this.cache.clear();
    debugTrace.loader(`Cleared shard cache: ${count} shards, ${formatBytes(bytes)} freed`);
  }

  /**
   * Configure cache size based on model type.
   * For MoE models, cache enough shards for 2x top-k experts + 1 dense shard.
   * For dense models, keep the default (2 shards).
   */
  configureForModel(manifest: RDRRManifest | null, hasCustomLoader: boolean): void {
    if (!manifest) return;
    this.manifest = manifest;

    const moe = manifest.moeConfig;
    if (moe && moe.numExpertsPerToken > 0) {
      // For MoE: cache 2x top-k experts (for current + next layer prefetch) + 1 dense shard
      const expertCacheSize = (moe.numExpertsPerToken * 2) + 1;
      // Cap at reasonable maximum (16 shards = ~1GB at 64MB/shard)
      this.maxEntries = Math.min(16, Math.max(4, expertCacheSize));
      debugTrace.loader(`MoE shard cache: ${this.maxEntries} entries (${moe.numExpertsPerToken} experts/token)`);
    } else if (hasCustomLoader) {
      // Network loading: use larger cache to avoid re-fetching shards.
      this.maxEntries = 16;
      debugTrace.loader(`Network shard cache: ${this.maxEntries} entries (avoiding re-fetch)`);
    } else {
      // OPFS (disk) loading - keep small cache, disk reads are fast
      this.maxEntries = 2;
    }
  }
}

/**
 * Create a new shard cache with default settings
 */
export function createShardCache(maxEntries = 2): ShardCache {
  return new ShardCache({ maxEntries });
}
