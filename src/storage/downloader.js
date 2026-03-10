

import {
  parseManifest,
  getManifestUrl,
} from '../formats/rdrr/index.js';

import {
  openModelStore,
  createFileWriter,
  createStreamingHasher,
  writeShard,
  shardExists,
  loadShard,
  loadFileFromStore,
  deleteShard,
  deleteFileFromStore,
  saveManifest,
  saveTokenizer,
  saveTokenizerModel,
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
  getDistributionConfig,
  getDefaultConcurrency,
  getMaxRetries,
  getInitialRetryDelayMs,
  getMaxRetryDelayMs,
  getProgressUpdateIntervalMs,
  getRequiredContentEncoding,
} from './download-types.js';

import { downloadShard as downloadShardFromDistribution } from '../distribution/shard-delivery.js';
import { resolveSourceArtifact, normalizeSourceArtifactPath } from './source-artifact-store.js';

// ============================================================================
// Module State
// ============================================================================


let db = null;

const activeDownloads = new Map();

function buildManifestVersionSet(manifest) {
  const sourceArtifact = resolveSourceArtifact(manifest);
  if (!manifest || typeof manifest !== 'object') return 'manifest:invalid';
  const shards = Array.isArray(manifest.shards)
    ? manifest.shards.map((shard, index) => ({
      index,
      filename: shard?.filename ?? null,
      size: shard?.size ?? null,
      hash: shard?.hash ?? null,
    }))
    : [];
  const payload = {
    modelId: manifest.modelId ?? null,
    version: manifest.version ?? null,
    hashAlgorithm: manifest.hashAlgorithm ?? null,
    tensorCount: manifest.tensorCount ?? null,
    totalSize: manifest.totalSize ?? null,
    shards,
    sourceRuntime: sourceArtifact?.sourceRuntime ?? null,
  };
  return JSON.stringify(payload);
}

function createDefaultSourceStats() {
  return {
    cache: 0,
    p2p: 0,
    http: 0,
    unknown: 0,
  };
}

function normalizeSourceStats(value) {
  const defaults = createDefaultSourceStats();
  if (!value || typeof value !== 'object') {
    return defaults;
  }
  return {
    cache: Number.isFinite(value.cache) ? Math.max(0, Number(value.cache)) : defaults.cache,
    p2p: Number.isFinite(value.p2p) ? Math.max(0, Number(value.p2p)) : defaults.p2p,
    http: Number.isFinite(value.http) ? Math.max(0, Number(value.http)) : defaults.http,
    unknown: Number.isFinite(value.unknown) ? Math.max(0, Number(value.unknown)) : defaults.unknown,
  };
}

function isTokenizerJsonRequired(tokenizer) {
  return Boolean(
    tokenizer
    && (tokenizer.type === 'bundled' || tokenizer.type === 'huggingface')
    && typeof tokenizer.file === 'string'
    && tokenizer.file.length > 0
  );
}

function getTokenizerModelPath(tokenizer) {
  if (!tokenizer || typeof tokenizer !== 'object') {
    return null;
  }
  const explicit = typeof tokenizer.sentencepieceModel === 'string'
    ? tokenizer.sentencepieceModel
    : null;
  if (explicit && explicit.length > 0) {
    return explicit;
  }
  if (tokenizer.type === 'sentencepiece') {
    return 'tokenizer.model';
  }
  return null;
}

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
            completedShards: new Set(result.completedShards),
            manifestVersionSet: typeof result.manifestVersionSet === 'string'
              ? result.manifestVersionSet
              : buildManifestVersionSet(result.manifest),
            sourceStats: normalizeSourceStats(result.sourceStats),
            lastSource: typeof result.lastSource === 'string' ? result.lastSource : null,
            lastSourcePath: typeof result.lastSourcePath === 'string' ? result.lastSourcePath : null,
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

function createAbortError(message = 'Download aborted') {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
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

function joinArtifactUrl(baseUrl, relativePath) {
  const root = String(baseUrl || '').trim();
  const rel = normalizeSourceArtifactPath(relativePath);
  if (!root || !rel) {
    throw new Error('joinArtifactUrl requires baseUrl and relativePath.');
  }
  return new URL(rel, root.endsWith('/') ? root : `${root}/`).href;
}

async function fileExistsInStore(path) {
  try {
    await loadFileFromStore(path);
    return true;
  } catch (error) {
    const message = String(error?.message || '');
    return error?.name === 'NotFoundError' || message.toLowerCase().includes('not found')
      ? false
      : Promise.reject(error);
  }
}

async function computeAssetHash(payload, algorithm = 'sha256') {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const hasher = await createStreamingHasher(String(algorithm || 'sha256').trim().toLowerCase());
  hasher.update(bytes);
  const digest = await hasher.finalize();
  return Array.from(digest)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function downloadSourceAsset(url, asset, options = {}) {
  const response = await fetchWithRetry(url, { signal: options.signal });
  const expectedSize = Number.isFinite(asset?.size) ? Math.max(0, Math.floor(Number(asset.size))) : null;
  const expectedHash = typeof asset?.hash === 'string' && asset.hash.trim() ? asset.hash.trim().toLowerCase() : null;
  const hashAlgorithm = typeof asset?.hashAlgorithm === 'string' && asset.hashAlgorithm.trim()
    ? asset.hashAlgorithm.trim().toLowerCase()
    : 'sha256';
  const writer = await createFileWriter(asset.path);
  const hasher = expectedHash ? await createStreamingHasher(hashAlgorithm) : null;
  let receivedBytes = 0;
  try {
    if (response.body && typeof response.body.getReader === 'function') {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (!(value instanceof Uint8Array) || value.byteLength <= 0) {
          continue;
        }
        await writer.write(value);
        hasher?.update(value);
        receivedBytes += value.byteLength;
        options.onProgress?.(receivedBytes);
      }
    } else {
      const bytes = new Uint8Array(await response.arrayBuffer());
      await writer.write(bytes);
      hasher?.update(bytes);
      receivedBytes += bytes.byteLength;
      options.onProgress?.(receivedBytes);
    }
    await writer.close();

    if (expectedSize != null && receivedBytes !== expectedSize) {
      throw new Error(
        `Asset size mismatch for ${asset.path}: expected ${expectedSize}, got ${receivedBytes}`
      );
    }

    if (hasher && expectedHash) {
      const computedHashBytes = await hasher.finalize();
      const computedHash = Array.from(computedHashBytes)
        .map((byte) => byte.toString(16).padStart(2, '0'))
        .join('');
      if (computedHash !== expectedHash) {
        throw new Error(
          `Asset hash mismatch for ${asset.path}: expected ${expectedHash}, got ${computedHash}`
        );
      }
    }

    return {
      source: 'http',
      path: asset.path,
      bytes: receivedBytes,
    };
  } catch (error) {
    try {
      await writer.abort();
    } catch {}
    try {
      await deleteFileFromStore(asset.path);
    } catch {}
    throw error;
  }
}


async function downloadShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  return downloadShardFromDistribution(baseUrl, shardIndex, shardInfo, {
    ...options,
    distributionConfig: getDistributionConfig(),
  });
}

export async function persistDownloadedShardIfNeeded(
  result,
  shardIndex,
  options = {}
) {
  const writeShardFn = typeof options.writeShardFn === 'function'
    ? options.writeShardFn
    : writeShard;

  if (!result || typeof result !== 'object') {
    throw new Error(`Shard ${shardIndex}: download result is missing`);
  }
  if (result.wrote === true) {
    return false;
  }
  if (result.source === 'cache') {
    return false;
  }
  if (!(result.buffer instanceof ArrayBuffer)) {
    throw new Error(`Shard ${shardIndex}: source "${result.source}" returned non-persisted data without buffer`);
  }
  await writeShardFn(shardIndex, result.buffer, { verify: false });
  return true;
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
    modelId: overrideModelId = undefined,
    signal: externalSignal = null,
  } = options;

  if (externalSignal?.aborted) {
    throw createAbortError();
  }

  // Request persistent storage if needed
  if (requestPersist) {
    await requestPersistence();
  }

  // Fetch and parse manifest
  const manifestUrl = getManifestUrl(baseUrl);
  const manifestResponse = await fetchWithRetry(manifestUrl);
  const manifestJson = await manifestResponse.text();
  const manifest = parseManifest(manifestJson);
  const directSourceArtifact = resolveSourceArtifact(manifest);
  const trackedShards = directSourceArtifact ? directSourceArtifact.sourceFiles : manifest.shards;
  const trackedTotalBytes = directSourceArtifact
    ? directSourceArtifact.totalBytes
    : manifest.totalSize;

  // Use override modelId for storage, or fall back to manifest's modelId
  const storageModelId = overrideModelId || manifest.modelId;

  // Check available space
  const spaceCheck = await checkSpaceAvailable(trackedTotalBytes);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(trackedTotalBytes, spaceCheck.info.available);
  }

  // Open model directory
  await openModelStore(storageModelId);

  // Check for existing download state
  const manifestVersionSet = buildManifestVersionSet(manifest);
  let state = await loadDownloadState(storageModelId);
  if (!state) {
    state = {
      modelId: storageModelId,
      baseUrl,
      manifest,
      manifestVersionSet,
      completedShards: new Set(),
      startTime: Date.now(),
      status: 'downloading',
      sourceStats: createDefaultSourceStats(),
      lastSource: null,
      lastSourcePath: null,
    };
  } else {
    state.status = 'downloading';
    const savedVersionSet = typeof state.manifestVersionSet === 'string'
      ? state.manifestVersionSet
      : buildManifestVersionSet(state.manifest);
    if (savedVersionSet !== manifestVersionSet) {
      log.warn('Downloader', `Manifest version-set changed for ${storageModelId}, resetting cached shards`);
      for (const idx of state.completedShards) {
        if (directSourceArtifact) {
          const sourceEntry = directSourceArtifact.sourceFiles[idx];
          if (sourceEntry) {
            await deleteFileFromStore(sourceEntry.path);
          }
        } else {
          await deleteShard(idx);
        }
      }
      state.completedShards.clear();
    }
    state.manifest = manifest;
    state.manifestVersionSet = manifestVersionSet;
    state.baseUrl = baseUrl;
    state.sourceStats = normalizeSourceStats(state.sourceStats);
    state.lastSource = typeof state.lastSource === 'string' ? state.lastSource : null;
    state.lastSourcePath = typeof state.lastSourcePath === 'string' ? state.lastSourcePath : null;
    // Check which shards actually exist (in case OPFS was cleared)
    for (const idx of state.completedShards) {
      if (directSourceArtifact) {
        const sourceEntry = directSourceArtifact.sourceFiles[idx];
        if (!sourceEntry || !(await fileExistsInStore(sourceEntry.path))) {
          state.completedShards.delete(idx);
        }
        continue;
      }
      if (!(await shardExists(idx))) {
        state.completedShards.delete(idx);
      }
    }
    // Verify hashes for completed shards; drop and re-download corrupt shards
    for (const idx of Array.from(state.completedShards)) {
      try {
        if (directSourceArtifact) {
          const sourceEntry = directSourceArtifact.sourceFiles[idx];
          if (!sourceEntry?.hash) {
            continue;
          }
          const payload = await loadFileFromStore(sourceEntry.path);
          const computedHash = await computeAssetHash(payload, sourceEntry.hashAlgorithm);
          if (computedHash !== sourceEntry.hash) {
            throw new Error(
              `Hash mismatch for source asset ${sourceEntry.path}: expected ${sourceEntry.hash}, got ${computedHash}`
            );
          }
        } else {
          await loadShard(idx, { verify: true });
        }
      } catch (err) {
        log.warn('Downloader', `Shard ${idx} failed verification, re-downloading`);
        state.completedShards.delete(idx);
        if (directSourceArtifact) {
          const sourceEntry = directSourceArtifact.sourceFiles[idx];
          if (sourceEntry) {
            await deleteFileFromStore(sourceEntry.path);
          }
        } else {
          await deleteShard(idx);
        }
      }
    }
  }

  // Create abort controller
  const abortController = new AbortController();
  const abortFromExternalSignal = () => {
    abortController.abort();
  };
  if (externalSignal && typeof externalSignal.addEventListener === 'function') {
    externalSignal.addEventListener('abort', abortFromExternalSignal, { once: true });
  }
  activeDownloads.set(storageModelId, {
    state,
    abortController
  });

  const totalShards = trackedShards.length;
  const requiredEncoding = getRequiredContentEncoding();
  
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
    const info = trackedShards[idx];
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
        totalBytes: trackedTotalBytes,
        downloadedBytes,
        percent: trackedTotalBytes > 0 ? (downloadedBytes / trackedTotalBytes) * 100 : 0,
        status:  (state).status,
        currentShard,
        speed: speedTracker.speed,
        lastSource: state.lastSource ?? null,
        lastSourcePath: state.lastSourcePath ?? null,
        sourceStats: normalizeSourceStats(state.sourceStats),
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
      if (directSourceArtifact) {
        const sourceAsset = directSourceArtifact.sourceFiles[shardIndex];
        if (!sourceAsset) {
          throw new Error(`Invalid source asset index: ${shardIndex}`);
        }
        const result = await downloadSourceAsset(
          joinArtifactUrl(baseUrl, sourceAsset.path),
          sourceAsset,
          {
            signal: abortController.signal,
            onProgress: (receivedBytes) => {
              const prev = shardProgress.get(shardIndex) || 0;
              const delta = Math.max(0, receivedBytes - prev);
              shardProgress.set(shardIndex, receivedBytes);
              downloadedBytes += delta;
              updateProgress(shardIndex);
            },
          }
        );

        const source = typeof result.source === 'string' ? result.source : 'unknown';
        const sourceStats = normalizeSourceStats(state.sourceStats);
        if (source in sourceStats) {
          sourceStats[source] += 1;
        } else {
          sourceStats.unknown += 1;
        }
        state.sourceStats = sourceStats;
        state.lastSource = source;
        state.lastSourcePath = typeof result.path === 'string' ? result.path : null;

        const observedBytes = shardProgress.get(shardIndex) || 0;
        const shardBytes = sourceAsset.size ?? result.bytes ?? observedBytes;
        if (shardBytes > observedBytes) {
          downloadedBytes += shardBytes - observedBytes;
        }
      } else {
        const shardInfo = manifest.shards[shardIndex];
        if (!shardInfo) {
          throw new Error(`Invalid shard index: ${shardIndex}`);
        }
        const algorithm = manifest.hashAlgorithm;
        if (!algorithm) {
          throw new Error('Manifest missing hashAlgorithm for download verification.');
        }
        const expectedHash = shardInfo.hash;
        if (!expectedHash) {
          throw new Error(`Shard ${shardIndex} is missing hash in manifest`);
        }
        const expectedSize = Number.isFinite(shardInfo.size) ? Math.floor(shardInfo.size) : null;
        const result = await downloadShard(baseUrl, shardIndex, shardInfo, {
          signal: abortController.signal,
          algorithm,
          requiredEncoding,
          expectedHash,
          expectedSize,
          expectedManifestVersionSet: manifestVersionSet,
          writeToStore: true,
          onProgress: ( p) => {
            const prev = shardProgress.get(shardIndex) || 0;
            const delta = Math.max(0, p.receivedBytes - prev);
            shardProgress.set(shardIndex, p.receivedBytes);
            downloadedBytes += delta;
            updateProgress(shardIndex);
          }
        });

        if (result.hash !== expectedHash) {
          await deleteShard(shardIndex);
          throw new Error(`Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${result.hash}`);
        }

        await persistDownloadedShardIfNeeded(result, shardIndex);

        const source = typeof result.source === 'string' ? result.source : 'unknown';
        const sourceStats = normalizeSourceStats(state.sourceStats);
        if (source in sourceStats) {
          sourceStats[source] += 1;
        } else {
          sourceStats.unknown += 1;
        }
        state.sourceStats = sourceStats;
        state.lastSource = source;
        state.lastSourcePath = typeof result.path === 'string' ? result.path : null;

        const observedBytes = shardProgress.get(shardIndex) || 0;
        const shardBytes = shardInfo.size ?? result.bytes ?? observedBytes;
        if (shardBytes > observedBytes) {
          downloadedBytes += shardBytes - observedBytes;
        }
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

    if (abortController.signal.aborted) {
      throw createAbortError();
    }

    // Verify all shards completed
    if (state.completedShards.size === totalShards) {
      state.status = 'completed';

      // Save manifest to OPFS
      await saveManifest(manifestJson);

      if (directSourceArtifact) {
        for (const asset of directSourceArtifact.auxiliaryFiles) {
          const alreadyPresent = await fileExistsInStore(asset.path);
          if (alreadyPresent) {
            continue;
          }
          await downloadSourceAsset(joinArtifactUrl(baseUrl, asset.path), asset, {
            signal: abortController.signal,
            onProgress: (receivedBytes) => {
              const previous = shardProgress.get(asset.path) || 0;
              const delta = Math.max(0, receivedBytes - previous);
              shardProgress.set(asset.path, receivedBytes);
              downloadedBytes += delta;
              updateProgress(null);
            },
          });
          const observedBytes = shardProgress.get(asset.path) || 0;
          shardProgress.delete(asset.path);
          const assetBytes = asset.size ?? observedBytes;
          if (assetBytes > observedBytes) {
            downloadedBytes += assetBytes - observedBytes;
          }
          updateProgress(null, true);
        }
      } else {
        // Download tokenizer assets if specified
        const tokenizer =  (manifest.tokenizer);
        if (isTokenizerJsonRequired(tokenizer)) {
          const tokenizerUrl = `${baseUrl}/${ (tokenizer).file}`;
          log.verbose('Downloader', `Fetching bundled tokenizer from ${tokenizerUrl}`);
          const tokenizerResponse = await fetchWithRetry(tokenizerUrl);
          const tokenizerJson = await tokenizerResponse.text();
          await saveTokenizer(tokenizerJson);
          log.verbose('Downloader', 'Saved bundled tokenizer.json');
        }

        const sentencepieceModel = getTokenizerModelPath(tokenizer);
        if (sentencepieceModel) {
          const modelUrl = `${baseUrl}/${sentencepieceModel}`;
          log.verbose('Downloader', `Fetching sentencepiece model from ${modelUrl}`);
          const modelResponse = await fetchWithRetry(modelUrl);
          const modelBuffer = await modelResponse.arrayBuffer();
          await saveTokenizerModel(modelBuffer);
          log.verbose('Downloader', 'Saved tokenizer.model');
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
    if (externalSignal && typeof externalSignal.removeEventListener === 'function') {
      externalSignal.removeEventListener('abort', abortFromExternalSignal);
    }
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

  return downloadModel(state.baseUrl, onProgress, {
    ...options,
    modelId: options.modelId ?? state.modelId,
  });
}


export async function getDownloadProgress(modelId) {
  // Check active downloads first
  const active = activeDownloads.get(modelId);
  if (active) {
    const state = active.state;
    const manifest = state.manifest;
    const directSourceArtifact = resolveSourceArtifact(manifest);
    const trackedShards = directSourceArtifact ? directSourceArtifact.sourceFiles : (manifest?.shards || []);
    const totalShards = trackedShards.length;

    let downloadedBytes = 0;
    for (const idx of state.completedShards) {
      const info = trackedShards[idx];
      if (info) downloadedBytes += info.size;
    }

    return {
      modelId,
      totalShards,
      completedShards: state.completedShards.size,
      totalBytes: directSourceArtifact ? directSourceArtifact.totalBytes : (manifest?.totalSize || 0),
      downloadedBytes,
      percent: manifest
        ? (
          downloadedBytes
          / (directSourceArtifact ? directSourceArtifact.totalBytes : manifest.totalSize || 1)
        ) * 100
        : 0,
      status: state.status,
      currentShard: null,
      speed: 0,
      lastSource: state.lastSource ?? null,
      lastSourcePath: state.lastSourcePath ?? null,
      sourceStats: normalizeSourceStats(state.sourceStats),
    };
  }

  // Check saved state
  const state = await loadDownloadState(modelId);
  if (!state) return null;
  const directSourceArtifact = resolveSourceArtifact(state.manifest);
  const trackedShards = directSourceArtifact ? directSourceArtifact.sourceFiles : state.manifest.shards;

  let downloadedBytes = 0;
  for (const idx of state.completedShards) {
    const shard = trackedShards[idx];
    if (shard) downloadedBytes += shard.size;
  }

  return {
    modelId,
    totalShards: trackedShards.length,
    completedShards: state.completedShards.size,
    totalBytes: directSourceArtifact ? directSourceArtifact.totalBytes : state.manifest.totalSize,
    downloadedBytes,
    percent: (
      downloadedBytes
      / (directSourceArtifact ? directSourceArtifact.totalBytes : state.manifest.totalSize || 1)
    ) * 100,
    status: state.status,
    currentShard: null,
    speed: 0,
    lastSource: state.lastSource ?? null,
    lastSourcePath: state.lastSourcePath ?? null,
    sourceStats: normalizeSourceStats(state.sourceStats),
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

  const directSourceArtifact = resolveSourceArtifact(state.manifest);
  const totalShards = directSourceArtifact ? directSourceArtifact.sourceFiles.length : state.manifest.shards.length;
  
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

  if (directSourceArtifact) {
    const missingAuxiliaryFiles = [];
    for (const entry of directSourceArtifact.auxiliaryFiles) {
      if (!(await fileExistsInStore(entry.path))) {
        missingAuxiliaryFiles.push(entry.path);
      }
    }
    if (missingAuxiliaryFiles.length > 0) {
      return {
        needed: true,
        reason: `Missing ${missingAuxiliaryFiles.length} direct-source auxiliary file(s)`,
        missingShards: [],
      };
    }
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
