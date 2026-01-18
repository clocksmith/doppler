

import {
  parseManifest,
  getManifestUrl,
} from './rdrr-format.js';

import {
  openModelStore,
  writeShard,
  shardExists,
  loadShard,
  deleteShard,
  saveManifest,
  saveTokenizer,
  createShardWriter,
  createStreamingHasher,
  computeHash,
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


let db = null;

const activeDownloads = new Map();

// ============================================================================
// IndexedDB Operations
// ============================================================================


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

    request.onupgradeneeded = ( event) => {
      const database =  (event.target).result;

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: 'modelId' });
        store.createIndex('status', 'status', { unique: false });
      }
    };
  });
}


async function saveDownloadState(state) {
  const database = await initDB();
  if (!database) return;

  try {
    await new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);

      
      const storeState = {
        ...state,
        completedShards: Array.from(state.completedShards)
      };

      const request = store.put(storeState);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to save download state'));
    });
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      db = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state save');
      return;
    }
    log.warn('Downloader', `Failed to save download state: ${ (error).message}`);
  }
}


async function loadDownloadState(modelId) {
  const database = await initDB();
  if (!database) return null;

  try {
    return await new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);

      const request = store.get(modelId);
      request.onsuccess = () => {
        const result =  (request.result);
        if (result) {
          
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
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      db = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state load');
      return null;
    }
    log.warn('Downloader', `Failed to load download state: ${ (error).message}`);
    return null;
  }
}


async function deleteDownloadState(modelId) {
  const database = await initDB();
  if (!database) return;

  try {
    await new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);

      const request = store.delete(modelId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to delete download state'));
    });
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      db = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state delete');
      return;
    }
    log.warn('Downloader', `Failed to delete download state: ${ (error).message}`);
  }
}

function isDatabaseClosingError(error) {
  const message =  (error)?.message ?? '';
  return message.includes('database connection is closing')
    ||  (error)?.name === 'InvalidStateError';
}

// ============================================================================
// Fetch Operations
// ============================================================================


async function fetchWithRetry(url, options = {}) {
  
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
      lastError =  (error);

      // Don't retry if aborted
      if ( (error).name === 'AbortError') {
        throw error;
      }

      // Don't retry on 4xx errors (except 429)
      if ( (error).message.includes('HTTP 4') && ! (error).message.includes('HTTP 429')) {
        throw error;
      }

      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, delay));
        delay = Math.min(delay * 2, getMaxRetryDelayMs());
      }
    }
  }

  throw  (lastError);
}


function buildShardUrl(baseUrl, shardInfo) {
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/${shardInfo.filename}`;
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

async function downloadShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  const { signal, onProgress, algorithm = 'blake3' } = options;
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
    const hash = await computeHash(buffer, algorithm);
    return { buffer, bytes: buffer.byteLength, hash, wrote: false };
  }

  const reader = response.body.getReader();
  const contentLength = shardInfo.size;
  let receivedBytes = 0;
  const hasher = await createStreamingHasher(algorithm);
  const writer = await createShardWriter(shardIndex);

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      hasher.update(value);
      await writer.write(value);
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

    const hashBytes = await hasher.finalize();
    const hash = bytesToHex(hashBytes);
    await writer.close();

    const elapsed = (performance.now() - startTime) / 1000;
    const speed = elapsed > 0 ? receivedBytes / elapsed : 0;
    const speedStr = formatBytes(speed) + '/s';
    log.verbose('Downloader', `Shard ${shardIndex}: network (${formatBytes(receivedBytes)}, ${elapsed.toFixed(2)}s @ ${speedStr})`);

    return { buffer: null, bytes: receivedBytes, hash, wrote: true };
  } catch (error) {
    await writer.abort();
    throw error;
  }
}

// ============================================================================
// Public API
// ============================================================================


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
  await openModelStore(storageModelId);

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

  
  const speedTracker = {
    lastBytes: downloadedBytes,
    lastTime: Date.now(),
    speed: 0
  };
  
  const shardProgress = new Map();
  let lastProgressUpdate = 0; // Throttle progress callbacks

  
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
        completedShards:  (state).completedShards.size,
        totalBytes: manifest.totalSize,
        downloadedBytes,
        percent: (downloadedBytes / manifest.totalSize) * 100,
        status:  (state).status,
        currentShard,
        speed: speedTracker.speed
      });
    }
  };

  // Download shards with concurrency control
  const downloadQueue = [...pendingShards];
  
  const inFlight = new Set();

  const downloadNext = async () => {
    if (downloadQueue.length === 0 || abortController.signal.aborted) {
      return;
    }

    const shardIndex =  (downloadQueue.shift());
    inFlight.add(shardIndex);
    updateProgress(shardIndex);

    try {
      const shardInfo = manifest.shards[shardIndex];
      if (!shardInfo) {
        throw new Error(`Invalid shard index: ${shardIndex}`);
      }
      const algorithm = manifest.hashAlgorithm || 'blake3';
      const result = await downloadShard(baseUrl, shardIndex, shardInfo, {
        signal: abortController.signal,
        algorithm,
        onProgress: ( p) => {
          const prev = shardProgress.get(shardIndex) || 0;
          const delta = Math.max(0, p.receivedBytes - prev);
          shardProgress.set(shardIndex, p.receivedBytes);
          downloadedBytes += delta;
          updateProgress(shardIndex);
        }
      });

      const expectedHash = shardInfo.hash || shardInfo.blake3;
      if (expectedHash && result.hash !== expectedHash) {
        await deleteShard(shardIndex);
        throw new Error(`Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${result.hash}`);
      }

      if (!result.wrote && result.buffer) {
        await writeShard(shardIndex, result.buffer, { verify: false });
      }

      // Update state
       (state).completedShards.add(shardIndex);
      shardProgress.delete(shardIndex);

      // Save progress
      await saveDownloadState( (state));
      updateProgress(null, true); // Force update on shard completion

    } catch (error) {
      if ( (error).name === 'AbortError') {
         (state).status = 'paused';
        await saveDownloadState( (state));
        throw error;
      }
      // Re-add to queue for retry (will be handled by next attempt)
      throw error;
    } finally {
      inFlight.delete(shardIndex);
    }
  };

  // Track errors from concurrent downloads
  
  const downloadErrors = [];

  try {
    // Process queue with concurrency limit
    
    const downloadPromises = new Set();

    while (downloadQueue.length > 0 || inFlight.size > 0) {
      if (abortController.signal.aborted) break;

      // Start new downloads up to concurrency limit
      while (inFlight.size < concurrency && downloadQueue.length > 0) {
        const promise = downloadNext().catch(( error) => {
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
      const tokenizer =  (manifest.tokenizer);
      const hasBundledTokenizer = (tokenizer?.type === 'bundled' || tokenizer?.type === 'huggingface') && tokenizer?.file;
      if (hasBundledTokenizer) {
        try {
          const tokenizerUrl = `${baseUrl}/${ (tokenizer).file}`;
          log.verbose('Downloader', `Fetching bundled tokenizer from ${tokenizerUrl}`);
          const tokenizerResponse = await fetchWithRetry(tokenizerUrl);
          const tokenizerJson = await tokenizerResponse.text();
          await saveTokenizer(tokenizerJson);
          log.verbose('Downloader', 'Saved bundled tokenizer.json');
        } catch (err) {
          log.warn('Downloader', `Failed to download tokenizer.json: ${ (err).message}`);
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
    state.error =  (error).message;
    await saveDownloadState(state);
    throw error;

  } finally {
    activeDownloads.delete(storageModelId);
  }
}


export function pauseDownload(modelId) {
  const download = activeDownloads.get(modelId);
  if (!download) return false;

  download.abortController.abort();
  return true;
}


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


export async function listDownloads() {
  const database = await initDB();
  if (!database) return [];

  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);

    const request = store.getAll();
    request.onsuccess = async () => {
      
      const results = [];
      for (const state of  (request.result)) {
        const progress = await getDownloadProgress(state.modelId);
        if (progress) results.push(progress);
      }
      resolve(results);
    };
    request.onerror = () => reject(new Error('Failed to list downloads'));
  });
}


export async function cancelDownload(modelId) {
  // Abort if active
  pauseDownload(modelId);

  // Remove state
  await deleteDownloadState(modelId);

  return true;
}


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


export function formatSpeed(bytesPerSecond) {
  return `${formatBytes(bytesPerSecond)}/s`;
}


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
