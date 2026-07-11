import {
  modelExists,
  openModelStore,
  loadManifestFromStore,
  saveManifest,
  deleteModel,
  verifyIntegrity,
  computeSHA256,
} from '../storage/shard-manager.js';
import { createOpfsArtifactStorageContext } from '../storage/artifact-storage-context.js';
import { downloadModel, estimateTimeRemaining, formatSpeed } from '../storage/downloader.js';
import { isOPFSAvailable, formatBytes } from '../storage/quota.js';
import { parseManifest, getManifestUrl } from '../formats/rdrr/index.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { cloneJsonValue } from '../utils/clone-json.js';
import { log } from '../debug/index.js';
import {
  resolveSourceArtifact,
  verifyStoredSourceArtifact,
} from '../storage/source-artifact-store.js';

const MODULE = 'OPFSCache';
let cacheOperationQueue = Promise.resolve();

function runCacheOperation(run) {
  const operation = cacheOperationQueue.then(run, run);
  cacheOperationQueue = operation.then(() => undefined, () => undefined);
  return operation;
}

function toErrorMessage(error) {
  if (error instanceof Error && typeof error.message === 'string' && error.message.length > 0) {
    return error.message;
  }
  return String(error);
}

function normalizeExpectedManifestHash(value) {
  const raw = value && typeof value === 'object'
    ? value.hex ?? value.hash ?? value.digest ?? ''
    : value;
  const normalized = String(raw || '').trim().toLowerCase().replace(/^sha256:/, '');
  if (!normalized) return null;
  if (!/^[0-9a-f]{64}$/.test(normalized)) {
    throw new Error('expectedManifestHash must be a SHA-256 hex digest.');
  }
  return normalized;
}

function manifestTotalBytes(manifest) {
  const declared = Number(manifest?.totalSize);
  if (Number.isFinite(declared) && declared >= 0) return Math.floor(declared);
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  return shards.reduce((total, shard) => {
    const size = Number(shard?.size);
    return total + (Number.isFinite(size) && size > 0 ? Math.floor(size) : 0);
  }, 0);
}

async function sha256Text(text) {
  return computeSHA256(new TextEncoder().encode(String(text || '')));
}

function normalizeShardDescriptor(shard) {
  return {
    filename: typeof shard?.filename === 'string' ? shard.filename : null,
    size: Number.isFinite(shard?.size) ? shard.size : null,
    hash: typeof shard?.hash === 'string' ? shard.hash : null,
  };
}

function hasSameShardSet(aManifest, bManifest) {
  const aShards = Array.isArray(aManifest?.shards) ? aManifest.shards : [];
  const bShards = Array.isArray(bManifest?.shards) ? bManifest.shards : [];
  if (aShards.length !== bShards.length) {
    return false;
  }
  for (let i = 0; i < aShards.length; i += 1) {
    const a = normalizeShardDescriptor(aShards[i]);
    const b = normalizeShardDescriptor(bShards[i]);
    if (a.filename !== b.filename || a.size !== b.size || a.hash !== b.hash) {
      return false;
    }
  }
  return true;
}

function preserveCachedSourceRuntimeMetadata(remoteManifest, cachedManifest) {
  const cachedSourceRuntime = cachedManifest?.metadata?.sourceRuntime;
  if (!cachedSourceRuntime || typeof cachedSourceRuntime !== 'object') {
    return {
      manifest: remoteManifest,
      changed: false,
    };
  }
  if (resolveSourceArtifact(remoteManifest)) {
    return {
      manifest: remoteManifest,
      changed: false,
    };
  }

  const mergedManifest = cloneJsonValue(remoteManifest);
  if (!mergedManifest || typeof mergedManifest !== 'object' || Array.isArray(mergedManifest)) {
    return {
      manifest: remoteManifest,
      changed: false,
    };
  }

  const metadata = (
    mergedManifest.metadata
    && typeof mergedManifest.metadata === 'object'
    && !Array.isArray(mergedManifest.metadata)
  )
    ? cloneJsonValue(mergedManifest.metadata)
    : {};
  metadata.sourceRuntime = cloneJsonValue(cachedSourceRuntime);
  mergedManifest.metadata = metadata;

  return {
    manifest: mergedManifest,
    changed: true,
  };
}

// buildManifestFingerprint compares a deliberate subset of manifest fields.
// Compared fields (partial match by design):
//   - modelId, modelHash, hashAlgorithm: identity and integrity algorithm
//   - quantization, quantizationInfo (weights/embeddings/compute/variantTag/layout):
//     determines weight format compatibility
//   - inference.layerPattern (type/globalPattern/period/offset/layerTypes):
//     determines layer dispatch structure
//   - shards (filename/size/hash per shard): data identity
//   - sourceArtifactFingerprint: tracks direct-source asset changes
//
// NOT compared (intentionally excluded because they change without affecting
// cached shard validity):
//   - manifest.version, manifest.config, manifest.architecture,
//     manifest.inference.execution, manifest.tokenizer, manifest.moeConfig,
//     manifest.inference.attention, manifest.inference.output
// This partial match avoids spurious re-downloads when only non-shard-affecting
// metadata changes. Cache hits still require exact manifest-text equality below;
// when text changes but shards do not, we refresh only the cached manifest.
function buildManifestFingerprint(manifest) {
  const sourceArtifactFingerprint = resolveSourceArtifact(manifest)?.fingerprint ?? null;
  const inference = manifest?.inference ?? {};
  const layerPattern = inference?.layerPattern ?? {};
  const quantizationInfo = manifest?.quantizationInfo ?? {};
  const shards = Array.isArray(manifest?.shards)
    ? manifest.shards.map(normalizeShardDescriptor)
    : [];
  return JSON.stringify({
    modelId: manifest?.modelId ?? null,
    modelHash: manifest?.modelHash ?? null,
    hashAlgorithm: manifest?.hashAlgorithm ?? null,
    quantization: manifest?.quantization ?? null,
    quantizationInfo: {
      weights: quantizationInfo.weights ?? null,
      embeddings: quantizationInfo.embeddings ?? null,
      compute: quantizationInfo.compute ?? null,
      variantTag: quantizationInfo.variantTag ?? null,
      layout: quantizationInfo.layout ?? null,
    },
    inference: {
      layerPattern: {
        type: layerPattern.type ?? null,
        globalPattern: layerPattern.globalPattern ?? null,
        period: layerPattern.period ?? null,
        offset: layerPattern.offset ?? null,
        layerTypes: Array.isArray(layerPattern.layerTypes)
          ? [...layerPattern.layerTypes]
          : null,
      },
    },
    shards,
    sourceArtifactFingerprint,
  });
}

async function fetchRemoteManifest(modelBaseUrl) {
  const manifestUrl = getManifestUrl(modelBaseUrl);
  const response = await fetch(manifestUrl, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`manifest fetch failed (${response.status})`);
  }
  const text = await response.text();
  return { text, manifest: parseManifest(text) };
}

async function loadCachedManifest(modelId) {
  await openModelStore(modelId);
  const text = await loadManifestFromStore();
  if (!text) {
    return { text: null, manifest: null };
  }
  return { text, manifest: parseManifest(text) };
}

async function verifyCachedArtifact(manifest) {
  if (resolveSourceArtifact(manifest)) {
    return verifyStoredSourceArtifact(manifest, { checkHashes: false });
  }
  return verifyIntegrity({ checkHashes: false });
}

async function resolvePinnedCacheHit(modelId, expectedManifestHash, onProgress) {
  if (!expectedManifestHash || !isOPFSAvailable() || !await modelExists(modelId)) {
    return null;
  }
  try {
    const cachedManifest = await loadCachedManifest(modelId);
    if (!cachedManifest.text || !cachedManifest.manifest) return null;
    if (await sha256Text(cachedManifest.text) !== expectedManifestHash) return null;
    const integrity = await verifyCachedArtifact(cachedManifest.manifest);
    if (!integrity.valid) return null;
    const totalBytes = manifestTotalBytes(cachedManifest.manifest);
    onProgress?.({
      stage: 'cache-hit',
      modelId,
      message: `Verified OPFS cache hit: ${modelId}`,
      percent: 100,
      totalBytes,
      downloadedBytes: totalBytes,
    });
    return {
      cached: true,
      fromCache: true,
      cacheState: 'verified-hit',
      modelId,
      error: null,
      manifestText: cachedManifest.text,
      manifestHash: expectedManifestHash,
      manifest: cachedManifest.manifest,
      totalBytes,
    };
  } catch (error) {
    log.warn(MODULE, `Pinned cache validation failed for "${modelId}": ${toErrorMessage(error)}`);
    return null;
  }
}

function buildDownloadProgress(progress) {
  if (!progress) return null;
  const totalBytes = Number.isFinite(progress.totalBytes) ? progress.totalBytes : 0;
  const downloadedBytes = Number.isFinite(progress.downloadedBytes) ? progress.downloadedBytes : 0;
  const speed = Number.isFinite(progress.speed) ? progress.speed : 0;
  const remainingBytes = Math.max(0, totalBytes - downloadedBytes);
  return {
    stage: 'downloading',
    modelId: progress.modelId || null,
    totalShards: Number.isFinite(progress.totalShards) ? progress.totalShards : 0,
    completedShards: Number.isFinite(progress.completedShards) ? progress.completedShards : 0,
    totalBytes,
    downloadedBytes,
    percent: Number.isFinite(progress.percent) ? progress.percent : 0,
    speed,
    speedFormatted: speed > 0 ? formatSpeed(speed) : '',
    totalFormatted: totalBytes > 0 ? formatBytes(totalBytes) : '',
    downloadedFormatted: downloadedBytes > 0 ? formatBytes(downloadedBytes) : '',
    eta: speed > 0 && remainingBytes > 0 ? estimateTimeRemaining(remainingBytes, speed) : '',
    message: buildDownloadStatusLine(progress, speed, remainingBytes),
  };
}

function buildDownloadStatusLine(progress, speed, remainingBytes) {
  const parts = [];
  const downloaded = Number.isFinite(progress.downloadedBytes) ? formatBytes(progress.downloadedBytes) : '0 B';
  const total = Number.isFinite(progress.totalBytes) ? formatBytes(progress.totalBytes) : '?';
  parts.push(`${downloaded} / ${total}`);
  if (Number.isFinite(progress.completedShards) && Number.isFinite(progress.totalShards)) {
    parts.push(`shard ${progress.completedShards}/${progress.totalShards}`);
  }
  if (speed > 0) {
    parts.push(formatSpeed(speed));
  }
  if (speed > 0 && remainingBytes > 0) {
    parts.push(`~${estimateTimeRemaining(remainingBytes, speed)} remaining`);
  }
  return parts.join(' | ');
}

async function ensureModelCachedUnlocked(modelId, modelBaseUrl, onProgress = null) {
  if (!modelId || !modelBaseUrl) {
    return {
      cached: false,
      fromCache: false,
      cacheState: 'error',
      modelId,
      error: 'missing-args',
    };
  }

  if (!isOPFSAvailable()) {
    log.warn(MODULE, 'OPFS not available in this browser');
    return {
      cached: false,
      fromCache: false,
      cacheState: 'error',
      modelId,
      error: 'opfs-unavailable',
    };
  }

  let needsFullImport = false;

  try {
    const exists = await modelExists(modelId);
    if (exists) {
      try {
        const [{ text: remoteManifestText, manifest: remoteManifest }, { text: cachedManifestText, manifest: cachedManifest }] = await Promise.all([
          fetchRemoteManifest(modelBaseUrl),
          loadCachedManifest(modelId),
        ]);

        if (!cachedManifestText || !cachedManifest) {
          log.warn(MODULE, `Cache miss: "${modelId}" has no readable manifest in OPFS; re-importing`);
          needsFullImport = true;
        } else {
          const cachedSourceArtifact = resolveSourceArtifact(cachedManifest);
          const sourceIntegrity = cachedSourceArtifact
            ? await verifyStoredSourceArtifact(cachedManifest, { checkHashes: false })
            : null;
          const sourceIntegrityValid = !sourceIntegrity || sourceIntegrity.valid;
          if (sourceIntegrity && !sourceIntegrity.valid) {
            log.warn(
              MODULE,
              `Cache stale: "${modelId}" direct-source assets are incomplete (${sourceIntegrity.missingFiles.join(', ')})`
            );
          }
          const cachedFingerprint = buildManifestFingerprint(cachedManifest);
          const remoteFingerprint = buildManifestFingerprint(remoteManifest);
          const manifestTextMatches = cachedManifestText === remoteManifestText;
          if (sourceIntegrityValid && manifestTextMatches && cachedFingerprint === remoteFingerprint) {
            const shardIntegrity = cachedSourceArtifact
              ? sourceIntegrity
              : await verifyCachedArtifact(cachedManifest);
            if (shardIntegrity.valid) {
              log.info(MODULE, `Cache hit: "${modelId}"`);
              onProgress?.({ stage: 'cache-hit', modelId, message: `OPFS cache hit: ${modelId}`, percent: 100 });
              return {
                cached: true,
                fromCache: true,
                cacheState: 'hit',
                modelId,
                error: null,
              };
            }
            log.warn(
              MODULE,
              `Cache incomplete: "${modelId}" is missing shards ${shardIntegrity.missingShards.join(', ')}`
            );
            needsFullImport = true;
          }

          if (!needsFullImport) {
            const sameShards = hasSameShardSet(cachedManifest, remoteManifest);
            const sameHashAlgorithm = (cachedManifest?.hashAlgorithm ?? null) === (remoteManifest?.hashAlgorithm ?? null);
            if (sourceIntegrityValid && sameShards && sameHashAlgorithm) {
              const preservedManifest = preserveCachedSourceRuntimeMetadata(remoteManifest, cachedManifest);
              const manifestTextToSave = preservedManifest.changed
                ? JSON.stringify(preservedManifest.manifest)
                : remoteManifestText;
              await openModelStore(modelId);
              await saveManifest(manifestTextToSave);
              const refreshMessage = preservedManifest.changed
                ? `Manifest refreshed: ${modelId} (shards unchanged, preserved direct-source metadata)`
                : `Manifest refreshed: ${modelId} (shards unchanged)`;
              log.info(MODULE, `Cache manifest refreshed: "${modelId}"${preservedManifest.changed ? ' (preserved direct-source metadata)' : ' (shards unchanged)'}`);
              onProgress?.({ stage: 'cache-refresh', modelId, message: refreshMessage, percent: 100 });
              return {
                cached: true,
                fromCache: false,
                cacheState: 'manifest-refresh',
                modelId,
                error: null,
              };
            }
          }

          log.info(MODULE, `Cache stale: "${modelId}" manifest/shards changed; deleting old version and re-importing`);
          onProgress?.({ stage: 'cache-invalidate', modelId, message: `Purging stale OPFS cache for ${modelId}`, percent: 0 });
          try {
            await deleteModel(modelId);
            log.info(MODULE, `Deleted stale OPFS cache for "${modelId}"`);
          } catch (deleteError) {
            log.warn(MODULE, `Failed to delete stale cache for "${modelId}": ${toErrorMessage(deleteError)}`);
          }
          needsFullImport = true;
        }
      } catch (error) {
        const message = toErrorMessage(error);
        log.warn(MODULE, `Cache validation failed (${message}); refusing cached model "${modelId}"`);
        return {
          cached: false,
          fromCache: false,
          cacheState: 'error',
          modelId,
          error: message,
        };
      }
    }
  } catch (error) {
    const message = toErrorMessage(error);
    log.warn(MODULE, `Cache check failed: ${message}`);
    return {
      cached: false,
      fromCache: false,
      cacheState: 'error',
      modelId,
      error: message,
    };
  }

  if (!needsFullImport) {
    log.info(MODULE, `Cache miss: "${modelId}". Triggering full model download from ${modelBaseUrl}`);
  }

  onProgress?.({ stage: 'download-start', modelId, message: `Downloading ${modelId}...`, percent: 0 });

  try {
    const success = await downloadModel(modelBaseUrl, (progress) => {
      if (!progress) return;
      const enriched = buildDownloadProgress(progress);
      if (enriched) {
        onProgress?.(enriched);
      }
      const shard = Number.isFinite(progress.completedShards) ? progress.completedShards : '?';
      const total = Number.isFinite(progress.totalShards) ? progress.totalShards : '?';
      const mb = Number.isFinite(progress.downloadedBytes)
        ? (progress.downloadedBytes / (1024 * 1024)).toFixed(1)
        : '?';
      log.verbose(MODULE, `Shard ${shard}/${total} (${mb} MB)`);
    });

    if (success) {
      log.info(MODULE, `Import complete: "${modelId}"`);
      onProgress?.({ stage: 'download-complete', modelId, message: `Download complete: ${modelId}`, percent: 100 });
      return {
        cached: true,
        fromCache: false,
        cacheState: 'imported',
        modelId,
        error: null,
      };
    }
    return {
      cached: false,
      fromCache: false,
      cacheState: 'error',
      modelId,
      error: 'download-incomplete',
    };
  } catch (error) {
    const message = toErrorMessage(error);
    log.error(MODULE, `Import failed: ${message}`);
    return {
      cached: false,
      fromCache: false,
      cacheState: 'error',
      modelId,
      error: message,
    };
  }
}

export function ensureModelCached(modelId, modelBaseUrl, onProgress = null) {
  return runCacheOperation(() => ensureModelCachedUnlocked(modelId, modelBaseUrl, onProgress));
}

export function ensureModelCachedSource(modelId, modelBaseUrl, onProgress = null, options = {}) {
  return runCacheOperation(async () => {
    const expectedManifestHash = normalizeExpectedManifestHash(options.expectedManifestHash);
    const pinnedHit = await resolvePinnedCacheHit(modelId, expectedManifestHash, onProgress);
    const cache = pinnedHit ?? await ensureModelCachedUnlocked(modelId, modelBaseUrl, onProgress);
    if (!cache.cached) {
      throw new Error(`Persistent model cache failed for "${modelId}": ${cache.error || cache.cacheState}`);
    }
    const cachedManifest = cache.manifest && cache.manifestText
      ? { manifest: cache.manifest, text: cache.manifestText }
      : await loadCachedManifest(modelId);
    if (!cachedManifest.text || !cachedManifest.manifest) {
      throw new Error(`Persistent model cache for "${modelId}" has no readable manifest.`);
    }
    const manifestHash = await sha256Text(cachedManifest.text);
    if (expectedManifestHash && manifestHash !== expectedManifestHash) {
      await deleteModel(modelId).catch(() => {});
      throw new Error(`Persistent model cache manifest hash mismatch for "${modelId}".`);
    }
    const runtime = getRuntimeConfig();
    const opfsPath = runtime.loading.opfsPath;
    const opfs = runtime.loading.storage.backend.opfs;
    const storageContext = await createOpfsArtifactStorageContext(modelId, cachedManifest.manifest, {
      opfsRootDir: opfsPath.opfsRootDir,
      useSyncAccessHandle: opfs.useSyncAccessHandle,
      maxConcurrentHandles: opfs.maxConcurrentHandles,
      verifyHashes: false,
      hashesTrusted: true,
    });
    return {
      ...cache,
      manifest: cachedManifest.manifest,
      manifestText: cachedManifest.text,
      manifestHash,
      storageContext,
      storageBackend: 'opfs',
      totalBytes: cache.totalBytes ?? manifestTotalBytes(cachedManifest.manifest),
    };
  });
}
