/**
 * downloader.ts - Resumable Model Downloader
 *
 * Handles:
 * - Chunked downloads with progress reporting
 * - Resume support via IndexedDB state tracking
 * - Parallel shard downloads with concurrency control
 * - Automatic retry with exponential backoff
 * - Quota checking before downloads
 *
 * @module storage/downloader
 */

import {
  parseManifest,
  getManifestUrl,
} from './rdrr-format.js';

import {
  openModelDirectory,
  writeShard,
  shardExists,
  loadShard,
  deleteShard,
  saveManifest,
  saveTokenizer,
} from './shard-manager.js';

import {
  checkSpaceAvailable,
  QuotaExceededError,
  requestPersistence,
  formatBytes,
  isIndexedDBAvailable,
} from './quota.js';

import { log } from '../debug/index.js';

import {
  DB_NAME,
  DB_VERSION,
  STORE_NAME,
  getDefaultConcurrency,
  getMaxRetries,
  getInitialRetryDelayMs,
  getMaxRetryDelayMs,
  getProgressUpdateIntervalMs,
} from './download-types.js';

// ============================================================================
// Module State
// ============================================================================

/** @type {IDBDatabase | null} */
let db = null;
/** @type {Map<string, import('./download-types.js').ActiveDownload>} */
const activeDownloads = new Map();

// ============================================================================
// IndexedDB Operations
// ============================================================================

/**
 * Initializes the IndexedDB for download state persistence
 * @returns {Promise<IDBDatabase | null>}
 */
async function initDB() {
  if (db) return db;

  if (!isIndexedDBAvailable()) {
    log.warn('Downloader', 'IndexedDB unavailable, download resume will not work');
    return null;
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(new Error('Failed to open IndexedDB'));

    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };

    request.onupgradeneeded = (/** @type {IDBVersionChangeEvent} */ event) => {
      const database = /** @type {IDBOpenDBRequest} */ (event.target).result;

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: 'modelId' });
        store.createIndex('status', 'status', { unique: false });
      }
    };
  });
}

/**
 * Saves download state to IndexedDB
 * @param {import('./download-types.js').DownloadState} state
 * @returns {Promise<void>}
 */
async function saveDownloadState(state) {
  const database = await initDB();
  if (!database) return;

  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    /** @type {import('./download-types.js').SerializedDownloadState} */
    const storeState = {
      ...state,
      completedShards: Array.from(state.completedShards)
    };

    const request = store.put(storeState);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(new Error('Failed to save download state'));
  });
}

/**
 * Loads download state from IndexedDB
 * @param {string} modelId
 * @returns {Promise<import('./download-types.js').DownloadState | null>}
 */
async function loadDownloadState(modelId) {
  const database = await initDB();
  if (!database) return null;

  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const request = store.get(modelId);
    request.onsuccess = () => {
      const result = /** @type {import('./download-types.js').SerializedDownloadState | undefined} */ (request.result);
      if (result) {
        /** @type {import('./download-types.js').DownloadState} */
        const state = {
          ...result,
          completedShards: new Set(result.completedShards)
        };
        resolve(state);
      } else {
        resolve(null);
      }
    };
    request.onerror = () => reject(new Error('Failed to load download state'));
  });
}

/**
 * Deletes download state from IndexedDB
 * @param {string} modelId
 * @returns {Promise<void>}
 */
async function deleteDownloadState(modelId) {
  const database = await initDB();
  if (!database) return;

  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    const request = store.delete(modelId);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(new Error('Failed to delete download state'));
  });
}

// ============================================================================
// Fetch Operations
// ============================================================================

/**
 * Fetches data with retry logic
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<Response>}
 */
async function fetchWithRetry(url, options = {}) {
  /** @type {Error | undefined} */
  let lastError;
  const maxRetries = getMaxRetries();
  let delay = getInitialRetryDelayMs();

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        signal: options.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      lastError = /** @type {Error} */ (error);

      // Don't retry if aborted
      if (/** @type {Error} */ (error).name === 'AbortError') {
        throw error;
      }

      // Don't retry on 4xx errors (except 429)
      if (/** @type {Error} */ (error).message.includes('HTTP 4') && !/** @type {Error} */ (error).message.includes('HTTP 429')) {
        throw error;
      }

      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, delay));
        delay = Math.min(delay * 2, getMaxRetryDelayMs());
      }
    }
  }

  throw /** @type {Error} */ (lastError);
}

/**
 * @param {string} baseUrl
 * @param {import('./rdrr-format.js').ShardInfo} shardInfo
 * @returns {string}
 */
function buildShardUrl(baseUrl, shardInfo) {
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/${shardInfo.filename}`;
}

/**
 * Downloads a single shard
 * @param {string} baseUrl
 * @param {number} shardIndex
 * @param {import('./rdrr-format.js').ShardInfo} shardInfo
 * @param {{ signal?: AbortSignal; onProgress?: (p: import('./download-types.js').ShardProgress) => void }} [options]
 * @returns {Promise<ArrayBuffer>}
 */
async function downloadShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  const { signal, onProgress } = options;
  const startTime = performance.now();

  const url = buildShardUrl(baseUrl, shardInfo);
  const response = await fetchWithRetry(url, { signal });

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    const percent = shardInfo.size > 0
      ? Math.min(1, buffer.byteLength / shardInfo.size)
      : 1;
    onProgress?.({
      shardIndex,
      receivedBytes: buffer.byteLength,
      totalBytes: shardInfo.size,
      percent,
    });
    return buffer;
  }

  // Stream the response for progress tracking
  const reader = response.body.getReader();
  const contentLength = shardInfo.size;

  /** @type {Uint8Array[]} */
  const chunks = [];
  let receivedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    chunks.push(value);
    receivedBytes += value.length;

    if (onProgress) {
      onProgress({
        shardIndex,
        receivedBytes,
        totalBytes: contentLength,
        percent: (receivedBytes / contentLength) * 100
      });
    }
  }

  // Combine chunks into single buffer
  const buffer = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  const elapsed = (performance.now() - startTime) / 1000;
  const speed = elapsed > 0 ? receivedBytes / elapsed : 0;
  const speedStr = formatBytes(speed) + '/s';
  log.verbose('Downloader', `Shard ${shardIndex}: network (${formatBytes(receivedBytes)}, ${elapsed.toFixed(2)}s @ ${speedStr})`);

  return buffer.buffer;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Downloads a model with progress reporting and resume support
 * @param {string} baseUrl
 * @param {import('./download-types.js').ProgressCallback} [onProgress]
 * @param {import('./download-types.js').DownloadOptions} [options]
 * @returns {Promise<boolean>}
 */
export async function downloadModel(
  baseUrl,
  onProgress,
  options = {}
) {
  const {
    concurrency = getDefaultConcurrency(),
    requestPersist = true,
    modelId: overrideModelId = undefined
  } = options;

  // Request persistent storage if needed
  if (requestPersist) {
    await requestPersistence();
  }

  // Fetch and parse manifest
  const manifestUrl = getManifestUrl(baseUrl);
  const manifestResponse = await fetchWithRetry(manifestUrl);
  const manifestJson = await manifestResponse.text();
  const manifest = parseManifest(manifestJson);

  // Use override modelId for storage, or fall back to manifest's modelId
  const storageModelId = overrideModelId || manifest.modelId;

  // Check available space
  const spaceCheck = await checkSpaceAvailable(manifest.totalSize);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(manifest.totalSize, spaceCheck.info.available);
  }

  // Open model directory
  await openModelDirectory(storageModelId);

  // Check for existing download state
  let state = await loadDownloadState(storageModelId);
  if (!state) {
    state = {
      modelId: storageModelId,
      baseUrl,
      manifest,
      completedShards: new Set(),
      startTime: Date.now(),
      status: 'downloading'
    };
  } else {
    state.status = 'downloading';
    // Check which shards actually exist (in case OPFS was cleared)
    for (const idx of state.completedShards) {
      if (!(await shardExists(idx))) {
        state.completedShards.delete(idx);
      }
    }
    // Verify hashes for completed shards; drop and re-download corrupt shards
    for (const idx of Array.from(state.completedShards)) {
      try {
        await loadShard(idx, { verify: true });
      } catch (err) {
        log.warn('Downloader', `Shard ${idx} failed verification, re-downloading`);
        state.completedShards.delete(idx);
        await deleteShard(idx);
      }
    }
  }

  // Create abort controller
  const abortController = new AbortController();
  activeDownloads.set(storageModelId, {
    state,
    abortController
  });

  const totalShards = manifest.shards.length;
  /** @type {number[]} */
  const pendingShards = [];

  // Find shards that need downloading
  for (let i = 0; i < totalShards; i++) {
    if (!state.completedShards.has(i)) {
      pendingShards.push(i);
    }
  }

  // Progress tracking
  let downloadedBytes = 0;
  for (const idx of state.completedShards) {
    const info = manifest.shards[idx];
    if (info) downloadedBytes += info.size;
  }

  /** @type {import('./download-types.js').SpeedTracker} */
  const speedTracker = {
    lastBytes: downloadedBytes,
    lastTime: Date.now(),
    speed: 0
  };
  /** @type {Map<number, number>} */
  const shardProgress = new Map();
  let lastProgressUpdate = 0; // Throttle progress callbacks

  /**
   * @param {number | null} currentShard
   * @param {boolean} [force]
   */
  const updateProgress = (currentShard, force = false) => {
    const now = Date.now();

    // Throttle progress updates (unless forced for completion events)
    if (!force && now - lastProgressUpdate < getProgressUpdateIntervalMs()) {
      return;
    }
    lastProgressUpdate = now;

    const timeDelta = (now - speedTracker.lastTime) / 1000;
    if (timeDelta >= 1) {
      speedTracker.speed = (downloadedBytes - speedTracker.lastBytes) / timeDelta;
      speedTracker.lastBytes = downloadedBytes;
      speedTracker.lastTime = now;
    }

    if (onProgress) {
      onProgress({
        modelId: storageModelId,
        manifest,
        totalShards,
        completedShards: /** @type {import('./download-types.js').DownloadState} */ (state).completedShards.size,
        totalBytes: manifest.totalSize,
        downloadedBytes,
        percent: (downloadedBytes / manifest.totalSize) * 100,
        status: /** @type {import('./download-types.js').DownloadState} */ (state).status,
        currentShard,
        speed: speedTracker.speed
      });
    }
  };

  // Download shards with concurrency control
  const downloadQueue = [...pendingShards];
  /** @type {Set<number>} */
  const inFlight = new Set();

  const downloadNext = async () => {
    if (downloadQueue.length === 0 || abortController.signal.aborted) {
      return;
    }

    const shardIndex = /** @type {number} */ (downloadQueue.shift());
    inFlight.add(shardIndex);
    updateProgress(shardIndex);

    try {
      const shardInfo = manifest.shards[shardIndex];
      if (!shardInfo) {
        throw new Error(`Invalid shard index: ${shardIndex}`);
      }
      const buffer = await downloadShard(baseUrl, shardIndex, shardInfo, {
        signal: abortController.signal,
        onProgress: (/** @type {import('./download-types.js').ShardProgress} */ p) => {
          // Update per-shard progress and global throughput
          const prev = shardProgress.get(shardIndex) || 0;
          const delta = Math.max(0, p.receivedBytes - prev);
          shardProgress.set(shardIndex, p.receivedBytes);
          downloadedBytes += delta;
          updateProgress(shardIndex);
        }
      });

      // Write shard to OPFS with verification
      await writeShard(shardIndex, buffer, { verify: true });

      // Update state
      /** @type {import('./download-types.js').DownloadState} */ (state).completedShards.add(shardIndex);
      shardProgress.delete(shardIndex);

      // Save progress
      await saveDownloadState(/** @type {import('./download-types.js').DownloadState} */ (state));
      updateProgress(null, true); // Force update on shard completion

    } catch (error) {
      if (/** @type {Error} */ (error).name === 'AbortError') {
        /** @type {import('./download-types.js').DownloadState} */ (state).status = 'paused';
        await saveDownloadState(/** @type {import('./download-types.js').DownloadState} */ (state));
        throw error;
      }
      // Re-add to queue for retry (will be handled by next attempt)
      throw error;
    } finally {
      inFlight.delete(shardIndex);
    }
  };

  // Track errors from concurrent downloads
  /** @type {Error[]} */
  const downloadErrors = [];

  try {
    // Process queue with concurrency limit
    /** @type {Set<Promise<void>>} */
    const downloadPromises = new Set();

    while (downloadQueue.length > 0 || inFlight.size > 0) {
      if (abortController.signal.aborted) break;

      // Start new downloads up to concurrency limit
      while (inFlight.size < concurrency && downloadQueue.length > 0) {
        const promise = downloadNext().catch((/** @type {Error} */ error) => {
          // Collect errors instead of swallowing them
          if (error.name !== 'AbortError') {
            downloadErrors.push(error);
            log.error('Downloader', `Shard download failed: ${error.message}`);
          }
        });
        downloadPromises.add(promise);
        promise.finally(() => downloadPromises.delete(promise));
      }

      // Wait a bit before checking again
      await new Promise(r => setTimeout(r, 100));
    }

    // Wait for any remaining downloads to complete
    await Promise.all([...downloadPromises]);

    // Verify all shards completed
    if (state.completedShards.size === totalShards) {
      state.status = 'completed';

      // Save manifest to OPFS
      await saveManifest(manifestJson);

      // Download and save tokenizer.json if bundled/huggingface tokenizer is specified
      const tokenizer = /** @type {{ type?: string; file?: string } | undefined} */ (manifest.tokenizer);
      const hasBundledTokenizer = (tokenizer?.type === 'bundled' || tokenizer?.type === 'huggingface') && tokenizer?.file;
      if (hasBundledTokenizer) {
        try {
          const tokenizerUrl = `${baseUrl}/${/** @type {{ file: string }} */ (tokenizer).file}`;
          log.verbose('Downloader', `Fetching bundled tokenizer from ${tokenizerUrl}`);
          const tokenizerResponse = await fetchWithRetry(tokenizerUrl);
          const tokenizerJson = await tokenizerResponse.text();
          await saveTokenizer(tokenizerJson);
          log.verbose('Downloader', 'Saved bundled tokenizer.json');
        } catch (err) {
          log.warn('Downloader', `Failed to download tokenizer.json: ${/** @type {Error} */ (err).message}`);
          // Non-fatal - model will fall back to HuggingFace tokenizer
        }
      }

      // Clean up download state
      await deleteDownloadState(storageModelId);

      updateProgress(null, true); // Force final update
      return true;
    }

    // If we have errors and not all shards completed, report them
    if (downloadErrors.length > 0) {
      const errorMessages = downloadErrors.map(e => e.message).join('; ');
      throw new Error(`Download incomplete: ${downloadErrors.length} shard(s) failed. Errors: ${errorMessages}`);
    }

    return false;

  } catch (error) {
    state.status = 'error';
    state.error = /** @type {Error} */ (error).message;
    await saveDownloadState(state);
    throw error;

  } finally {
    activeDownloads.delete(storageModelId);
  }
}

/**
 * Pauses an active download
 * @param {string} modelId
 * @returns {boolean}
 */
export function pauseDownload(modelId) {
  const download = activeDownloads.get(modelId);
  if (!download) return false;

  download.abortController.abort();
  return true;
}

/**
 * Resumes a paused download
 * @param {string} modelId
 * @param {import('./download-types.js').ProgressCallback} [onProgress]
 * @param {import('./download-types.js').DownloadOptions} [options]
 * @returns {Promise<boolean>}
 */
export async function resumeDownload(
  modelId,
  onProgress,
  options = {}
) {
  const state = await loadDownloadState(modelId);
  if (!state) {
    throw new Error(`No download state found for model: ${modelId}`);
  }

  return downloadModel(state.baseUrl, onProgress, options);
}

/**
 * Gets the download progress for a model
 * @param {string} modelId
 * @returns {Promise<import('./download-types.js').DownloadProgress | null>}
 */
export async function getDownloadProgress(modelId) {
  // Check active downloads first
  const active = activeDownloads.get(modelId);
  if (active) {
    const state = active.state;
    const manifest = state.manifest;
    const totalShards = manifest?.shards?.length || 0;

    let downloadedBytes = 0;
    for (const idx of state.completedShards) {
      const info = manifest?.shards?.[idx];
      if (info) downloadedBytes += info.size;
    }

    return {
      modelId,
      totalShards,
      completedShards: state.completedShards.size,
      totalBytes: manifest?.totalSize || 0,
      downloadedBytes,
      percent: manifest ? (downloadedBytes / manifest.totalSize) * 100 : 0,
      status: state.status,
      currentShard: null,
      speed: 0
    };
  }

  // Check saved state
  const state = await loadDownloadState(modelId);
  if (!state) return null;

  let downloadedBytes = 0;
  for (const idx of state.completedShards) {
    const shard = state.manifest.shards[idx];
    if (shard) downloadedBytes += shard.size;
  }

  return {
    modelId,
    totalShards: state.manifest.shards.length,
    completedShards: state.completedShards.size,
    totalBytes: state.manifest.totalSize,
    downloadedBytes,
    percent: (downloadedBytes / state.manifest.totalSize) * 100,
    status: state.status,
    currentShard: null,
    speed: 0
  };
}

/**
 * Lists all in-progress or paused downloads
 * @returns {Promise<import('./download-types.js').DownloadProgress[]>}
 */
export async function listDownloads() {
  const database = await initDB();
  if (!database) return [];

  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const request = store.getAll();
    request.onsuccess = async () => {
      /** @type {import('./download-types.js').DownloadProgress[]} */
      const results = [];
      for (const state of /** @type {import('./download-types.js').SerializedDownloadState[]} */ (request.result)) {
        const progress = await getDownloadProgress(state.modelId);
        if (progress) results.push(progress);
      }
      resolve(results);
    };
    request.onerror = () => reject(new Error('Failed to list downloads'));
  });
}

/**
 * Cancels and removes a download
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
export async function cancelDownload(modelId) {
  // Abort if active
  pauseDownload(modelId);

  // Remove state
  await deleteDownloadState(modelId);

  return true;
}

/**
 * Checks if a model needs downloading
 * @param {string} modelId
 * @returns {Promise<import('./download-types.js').DownloadNeededResult>}
 */
export async function checkDownloadNeeded(modelId) {
  const state = await loadDownloadState(modelId);

  if (!state) {
    return {
      needed: true,
      reason: 'Model not downloaded',
      missingShards: []
    };
  }

  const totalShards = state.manifest.shards.length;
  /** @type {number[]} */
  const missingShards = [];

  for (let i = 0; i < totalShards; i++) {
    if (!state.completedShards.has(i)) {
      missingShards.push(i);
    }
  }

  if (missingShards.length > 0) {
    return {
      needed: true,
      reason: `Missing ${missingShards.length} of ${totalShards} shards`,
      missingShards
    };
  }

  return {
    needed: false,
    reason: 'Model fully downloaded',
    missingShards: []
  };
}

/**
 * Formats download speed for display
 * @param {number} bytesPerSecond
 * @returns {string}
 */
export function formatSpeed(bytesPerSecond) {
  return `${formatBytes(bytesPerSecond)}/s`;
}

/**
 * Estimates remaining download time
 * @param {number} remainingBytes
 * @param {number} bytesPerSecond
 * @returns {string}
 */
export function estimateTimeRemaining(remainingBytes, bytesPerSecond) {
  if (bytesPerSecond <= 0) return 'Calculating...';

  const seconds = remainingBytes / bytesPerSecond;

  if (seconds < 60) {
    return `${Math.ceil(seconds)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.ceil(seconds / 60);
    return `${minutes}m`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.ceil((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
}
