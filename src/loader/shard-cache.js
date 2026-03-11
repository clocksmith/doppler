import {
  loadShard as loadShardFromStore,
  loadShardRange as loadShardRangeFromStore,
  streamShardRange as streamShardRangeFromStore,
  computeHash,
  getStorageBackendType,
} from '../storage/shard-manager.js';
import { getExpectedShardHash } from '../formats/rdrr/index.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

export class ShardCache {
  #cache = new Map();
  #maxEntries;
  #customLoader = null;
  #customRangeLoader = null;
  #customStreamLoader = null;
  #verifyHashes;
  #manifest = null;
  #loadingConfig;
  #fetchPromises = new Map();
  #maxConcurrentLoads = 0;
  #inFlightLoads = 0;
  #highPriorityQueue = [];
  #lowPriorityQueue = [];
  #epoch = 0;

  lastSource = null;

  constructor(config) {
    this.#maxEntries = config.maxEntries;
    this.#customLoader = config.customLoader ?? null;
    this.#customRangeLoader = config.customRangeLoader ?? null;
    this.#customStreamLoader = config.customStreamLoader ?? null;
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
    if (config.customRangeLoader !== undefined) {
      this.#customRangeLoader = config.customRangeLoader;
    }
    if (config.customStreamLoader !== undefined) {
      this.#customStreamLoader = config.customStreamLoader;
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

  setCustomLoader(loader, verify = true, options = {}) {
    this.#customLoader = loader;
    this.#customRangeLoader = typeof options.loadRange === 'function'
      ? options.loadRange
      : null;
    this.#customStreamLoader = typeof options.streamRange === 'function'
      ? options.streamRange
      : null;
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

  get hasCustomRangeLoader() {
    return this.#customRangeLoader !== null;
  }

  get hasCustomStreamLoader() {
    return this.#customStreamLoader !== null;
  }

  get canStreamRanges() {
    return this.#customStreamLoader !== null || this.#customRangeLoader !== null;
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
    const epoch = this.#epoch;

    // 1. Check cache first
    if (this.#cache.has(shardIndex)) {
      const cached = this.#cache.get(shardIndex);
      // Refresh LRU order
      this.#cache.delete(shardIndex);
      this.#cache.set(shardIndex, cached);
      this.#setLastSource('RAM', 0, 'full', 'cache');
      log.verbose('ShardCache', `Shard ${shardIndex}: RAM${sizeStr ? ` (${sizeStr})` : ''}`);
      return cached;
    }

    // 2. Check if fetch is already in-flight - deduplicate concurrent requests
    const inFlight = this.#fetchPromises.get(shardIndex);
    if (inFlight && inFlight.epoch === epoch) {
      log.verbose('ShardCache', `Shard ${shardIndex}: waiting for in-flight fetch`);
      return inFlight.promise;
    }

    // 3. Start the actual fetch and store the promise for deduplication
    const fetchPromise = this.#scheduleLoad(
      priority,
      epoch,
      () => this.#doLoad(shardIndex, sizeStr, epoch)
    );
    const fetchEntry = { epoch, promise: fetchPromise };
    this.#fetchPromises.set(shardIndex, fetchEntry);

    try {
      const result = await fetchPromise;
      return result;
    } finally {
      // Remove from in-flight map when done (success or error)
      if (this.#fetchPromises.get(shardIndex) === fetchEntry) {
        this.#fetchPromises.delete(shardIndex);
      }
    }
  }

  #toRangeOffset(value) {
    const normalized = Number(value);
    if (!Number.isFinite(normalized)) return 0;
    const offsetValue = Math.trunc(normalized);
    return offsetValue > 0 ? offsetValue : 0;
  }

  #setLastSource(source, elapsed, mode, path, fallback = 'none') {
    this.lastSource = {
      source,
      elapsed,
      mode,
      path,
      fallback,
    };
  }

  #isUnsupportedRangeOrStream(error) {
    const code = String(error?.code || '').toLowerCase();
    if (code.includes('unsupported') || code.includes('not_supported')) {
      return true;
    }
    const message = String(error?.message || '').toLowerCase();
    return message.includes('not supported') || message.includes('unsupported');
  }

  #toArrayBuffer(data) {
    if (data instanceof Uint8Array) {
      return (data.byteOffset === 0 && data.byteLength === data.buffer.byteLength)
        ? data.buffer
        : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    }
    if (data instanceof ArrayBuffer) {
      return data;
    }
    throw new Error('Custom shard loader must return ArrayBuffer or Uint8Array.');
  }

  #throwShortStreamRead(shardIndex, start, want, produced, path) {
    throw new Error(
      `Shard ${shardIndex} short stream read via ${path}: ` +
      `offset=${start}, expected=${want}, got=${produced}.`
    );
  }

  async loadRange(shardIndex, offset = 0, length = null, options = {}) {
    const start = this.#toRangeOffset(offset);
    const want = length == null ? null : this.#toRangeOffset(length);

    if (this.#cache.has(shardIndex)) {
      const cached = this.#cache.get(shardIndex);
      // Refresh LRU order
      this.#cache.delete(shardIndex);
      this.#cache.set(shardIndex, cached);
      this.#setLastSource('RAM', 0, 'range', 'cache');
      const view = new Uint8Array(cached);
      const end = want == null ? view.length : Math.min(view.length, start + want);
      // Return a compact ArrayBuffer (downstream expects independent buffers).
      return view.slice(start, end).buffer;
    }

    if (this.#customRangeLoader) {
      try {
        const rangeStart = performance.now();
        const rangeData = await this.#customRangeLoader(shardIndex, start, want);
        const elapsed = (performance.now() - rangeStart) / 1000;
        this.#setLastSource('custom', elapsed, 'range', 'custom-range');
        return this.#toArrayBuffer(rangeData);
      } catch (error) {
        const unsupported = this.#isUnsupportedRangeOrStream(error);
        if (!unsupported) {
          throw error;
        }
        if (!this.#customLoader) {
          const backendStart = performance.now();
          const data = await loadShardRangeFromStore(shardIndex, start, want, options);
          const elapsed = (performance.now() - backendStart) / 1000;
          const backend = getStorageBackendType() ?? 'storage';
          this.#setLastSource(
            backend,
            elapsed,
            'range',
            'backend-range',
            'custom_range_not_supported'
          );
          return data;
        }
      }
    }

    if (this.#customLoader) {
      // Custom loaders without a range API: full-shard load, then slice.
      const full = await this.load(shardIndex, options);
      const view = new Uint8Array(full);
      const end = want == null ? view.length : Math.min(view.length, start + want);
      const fallbackTag = this.#customRangeLoader
        ? 'custom_range_not_supported'
        : 'custom_range_unavailable';
      this.#setLastSource('custom', this.lastSource?.elapsed ?? 0, 'range', 'custom-loader-slice', fallbackTag);
      return view.slice(start, end).buffer;
    }

    // Direct backend range read (no shard cache population).
    const backendStart = performance.now();
    const data = await loadShardRangeFromStore(shardIndex, start, want, options);
    const elapsed = (performance.now() - backendStart) / 1000;
    const backend = getStorageBackendType() ?? 'storage';
    this.#setLastSource(backend, elapsed, 'range', 'backend-range');
    return data;
  }

  async *streamRange(shardIndex, offset = 0, length = null, options = {}) {
    const start = this.#toRangeOffset(offset);
    const want = length == null ? null : this.#toRangeOffset(length);
    const chunkBytesRaw = Number(options?.chunkBytes);
    const chunkBytes = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
      ? Math.floor(chunkBytesRaw)
      : 4 * 1024 * 1024;

    if (this.#cache.has(shardIndex)) {
      const cached = this.#cache.get(shardIndex);
      this.#cache.delete(shardIndex);
      this.#cache.set(shardIndex, cached);
      this.#setLastSource('RAM', 0, 'stream', 'cache');
      const view = new Uint8Array(cached);
      const end = want == null ? view.length : Math.min(view.length, start + want);
      let produced = 0;
      for (let cursor = start; cursor < end; cursor += chunkBytes) {
        const sliceEnd = Math.min(end, cursor + chunkBytes);
        const chunk = view.slice(cursor, sliceEnd);
        produced += chunk.byteLength;
        yield chunk;
      }
      if (want != null && produced < want) {
        this.#throwShortStreamRead(shardIndex, start, want, produced, 'cache');
      }
      return;
    }

    if (this.#customStreamLoader) {
      const streamStart = performance.now();
      let produced = 0;
      try {
        for await (const chunk of this.#customStreamLoader(shardIndex, start, want, { chunkBytes })) {
          const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(this.#toArrayBuffer(chunk));
          if (bytes.byteLength > 0) {
            produced += bytes.byteLength;
            yield bytes;
          }
        }
      } catch (error) {
        const unsupported = this.#isUnsupportedRangeOrStream(error);
        if (!this.#customRangeLoader || (!unsupported && produced <= 0)) {
          throw error;
        }
        const fallbackReason = unsupported ? 'custom_stream_not_supported' : 'custom_stream_interrupted';
        let resumed = 0;
        let emptyRetries = 0;
        while (true) {
          const remaining = want == null ? chunkBytes : Math.max(0, want - produced - resumed);
          if (want != null && remaining <= 0) break;
          const requestLength = Math.min(chunkBytes, remaining);
          const chunk = await this.#customRangeLoader(
            shardIndex,
            start + produced + resumed,
            requestLength
          );
          const bytes = new Uint8Array(this.#toArrayBuffer(chunk));
          if (bytes.byteLength === 0) {
            if (emptyRetries < 1) {
              emptyRetries++;
              continue;
            }
            break;
          }
          emptyRetries = 0;
          resumed += bytes.byteLength;
          yield bytes;
        }
        if (want != null && produced + resumed < want) {
          this.#throwShortStreamRead(
            shardIndex,
            start,
            want,
            produced + resumed,
            'custom-range-fallback'
          );
        }
        const elapsed = (performance.now() - streamStart) / 1000;
        this.#setLastSource(
          'custom',
          elapsed,
          'stream',
          'custom-range-fallback',
          `${fallbackReason}${resumed > 0 ? '_resume' : ''}`
        );
        return;
      }

      if (want != null && produced < want && this.#customRangeLoader) {
        // Deterministic fallback: resume remaining bytes with the range loader.
        let resumed = 0;
        let emptyRetries = 0;
        while (produced + resumed < want) {
          const remaining = want - produced - resumed;
          const requestLength = Math.min(chunkBytes, remaining);
          const chunk = await this.#customRangeLoader(
            shardIndex,
            start + produced + resumed,
            requestLength
          );
          const bytes = new Uint8Array(this.#toArrayBuffer(chunk));
          if (bytes.byteLength === 0) {
            if (emptyRetries < 1) {
              emptyRetries++;
              continue;
            }
            break;
          }
          emptyRetries = 0;
          resumed += bytes.byteLength;
          yield bytes;
        }
        if (produced + resumed < want) {
          this.#throwShortStreamRead(
            shardIndex,
            start,
            want,
            produced + resumed,
            'custom-range-fallback'
          );
        }
        const elapsed = (performance.now() - streamStart) / 1000;
        this.#setLastSource(
          'custom',
          elapsed,
          'stream',
          'custom-range-fallback',
          'custom_stream_partial_resume'
        );
        return;
      }

      if (want != null && produced < want) {
        this.#throwShortStreamRead(shardIndex, start, want, produced, 'custom-stream');
      }
      const elapsed = (performance.now() - streamStart) / 1000;
      this.#setLastSource('custom', elapsed, 'stream', 'custom-stream');
      return;
    }

    if (this.#customRangeLoader) {
      const rangeStart = performance.now();
      let partialRetryUsed = false;
      let emptyRetries = 0;
      let produced = 0;
      while (true) {
        const remaining = want == null ? chunkBytes : Math.max(0, want - produced);
        if (want != null && remaining <= 0) break;
        const requestLength = Math.min(chunkBytes, remaining);
        const chunk = await this.#customRangeLoader(shardIndex, start + produced, requestLength);
        const bytes = new Uint8Array(this.#toArrayBuffer(chunk));
        if (bytes.byteLength === 0) {
          if (emptyRetries < 1) {
            emptyRetries++;
            partialRetryUsed = true;
            continue;
          }
          break;
        }
        emptyRetries = 0;
        produced += bytes.byteLength;
        yield bytes;
        if (bytes.byteLength < requestLength) {
          partialRetryUsed = true;
          if (want == null) {
            break;
          }
        }
      }
      if (want != null && produced < want) {
        this.#throwShortStreamRead(shardIndex, start, want, produced, 'custom-range');
      }
      this.#setLastSource(
        'custom',
        (performance.now() - rangeStart) / 1000,
        'stream',
        'custom-range',
        partialRetryUsed ? 'custom_range_partial_retry' : 'none'
      );
      return;
    }

    const streamStart = performance.now();
    let produced = 0;
    for await (const chunk of streamShardRangeFromStore(shardIndex, start, want, { chunkBytes })) {
      const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
      produced += bytes.byteLength;
      yield bytes;
    }
    if (want != null && produced < want) {
      this.#throwShortStreamRead(shardIndex, start, want, produced, 'backend-stream');
    }
    const elapsed = (performance.now() - streamStart) / 1000;
    const backend = getStorageBackendType() ?? 'storage';
    this.#setLastSource(backend, elapsed, 'stream', 'backend-stream');
  }

  prefetch(shardIndex) {
    return this.load(shardIndex, { priority: 'low' });
  }

  async #doLoad(shardIndex, sizeStr, epoch) {
    if (this.#customLoader) {
      const startTime = performance.now();
      let data = await this.#customLoader(shardIndex);

      // Verify hash if enabled
      if (this.#verifyHashes && this.#manifest) {
        const shardInfo = this.#manifest.shards?.[shardIndex];
        const algorithm = shardInfo?.hashAlgorithm ?? this.#manifest.hashAlgorithm;
        const expectedHash = getExpectedShardHash(shardInfo, algorithm);
        if (!expectedHash) {
          throw new Error(`Shard ${shardIndex} missing hash in manifest.`);
        }
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
      const arrayBuffer = this.#toArrayBuffer(data);

      if (epoch === this.#epoch) {
        this.#add(shardIndex, arrayBuffer);
      }

      const elapsed = (performance.now() - startTime) / 1000;
      this.#setLastSource('custom', elapsed, 'full', 'custom-loader');
      log.verbose('ShardCache', `Shard ${shardIndex}: network (${sizeStr}, ${elapsed.toFixed(2)}s)`);
      return arrayBuffer;
    }

    const storageStart = performance.now();
    const data = await loadShardFromStore(shardIndex);
    if (epoch === this.#epoch) {
      this.#add(shardIndex, data);
    }
    const elapsed = (performance.now() - storageStart) / 1000;
    const backend = getStorageBackendType() ?? 'storage';
    this.#setLastSource(backend, elapsed, 'full', 'backend-full');
    log.verbose('ShardCache', `Shard ${shardIndex}: ${backend} (${sizeStr}, ${elapsed.toFixed(2)}s)`);
    return data;
  }

  async #scheduleLoad(priority, epoch, task) {
    const limit = this.#maxConcurrentLoads > 0
      ? this.#maxConcurrentLoads
      : Number.POSITIVE_INFINITY;

    if (this.#inFlightLoads < limit) {
      if (epoch !== this.#epoch) {
        throw new Error('Shard load invalidated by cache clear().');
      }
      this.#inFlightLoads++;
      try {
        return await task();
      } finally {
        this.#inFlightLoads--;
        this.#drainQueue();
      }
    }

    return new Promise((resolve, reject) => {
      const entry = { task, resolve, reject, epoch };
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
      if (entry.epoch !== this.#epoch) {
        entry.reject(new Error('Shard load invalidated by cache clear().'));
        continue;
      }

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
    this.#epoch++;
    const queued = [...this.#highPriorityQueue, ...this.#lowPriorityQueue];
    this.#highPriorityQueue = [];
    this.#lowPriorityQueue = [];
    this.#fetchPromises.clear();
    for (const entry of queued) {
      entry.reject(new Error('Shard load invalidated by cache clear().'));
    }
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
