

import {
  parseManifest,
  getExpectedShardHash,
  getManifestUrl,
} from '../formats/rdrr/index.js';

import {
  openModelStore,
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
} from './quota.js';

import { log } from '../debug/index.js';

import {
  getDefaultConcurrency,
  getProgressUpdateIntervalMs,
  getRequiredContentEncoding,
} from './download-types.js';

import { resolveSourceArtifact } from './source-artifact-store.js';
import {
  buildManifestVersionSet,
  computeAssetHash,
  createDefaultSourceStats,
  fileExistsInStore,
  getTokenizerModelPath,
  isTokenizerJsonRequired,
  normalizeSourceStats,
  persistDownloadedShardIfNeeded,
} from './download/integrity.js';
import { createAbortError, fetchWithRetry } from './download/retry.js';
import {
  downloadShard,
  downloadSourceAsset,
  joinArtifactUrl,
} from './download/transport.js';
import {
  deleteDownloadState,
  loadAllDownloadStates,
  loadDownloadState,
  saveDownloadState,
} from './download/state.js';

export { persistDownloadedShardIfNeeded } from './download/integrity.js';
export { inspectModelDownloadResume } from './download/resume-inspection.js';

// ============================================================================
// Module State
// ============================================================================


const activeDownloads = new Map();

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

  let verifiedDownloadedBytes = 0;
  for (const idx of state.completedShards) {
    const info = trackedShards[idx];
    if (info) verifiedDownloadedBytes += Number(info.size || 0);
  }
  const requiredDownloadBytes = Math.max(0, trackedTotalBytes - verifiedDownloadedBytes);
  const spaceCheck = await checkSpaceAvailable(requiredDownloadBytes);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(requiredDownloadBytes, spaceCheck.info.available);
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
        const expectedHash = getExpectedShardHash(shardInfo, algorithm);
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
  const results = [];
  for (const state of await loadAllDownloadStates()) {
    const progress = await getDownloadProgress(state.modelId);
    if (progress) results.push(progress);
  }
  return results;
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
