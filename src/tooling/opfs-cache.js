import {
  modelExists,
  openModelStore,
  loadManifestFromStore,
  saveManifest,
  deleteModel,
} from '../storage/shard-manager.js';
import { downloadModel, estimateTimeRemaining, formatSpeed } from '../storage/downloader.js';
import { isOPFSAvailable, formatBytes } from '../storage/quota.js';
import { parseManifest, getManifestUrl } from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  resolveSourceArtifact,
  verifyStoredSourceArtifact,
} from '../storage/source-artifact-store.js';

const MODULE = 'OPFSCache';

function toErrorMessage(error) {
  if (error instanceof Error && typeof error.message === 'string' && error.message.length > 0) {
    return error.message;
  }
  return String(error);
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

export async function ensureModelCached(modelId, modelBaseUrl, onProgress = null) {
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

          const sameShards = hasSameShardSet(cachedManifest, remoteManifest);
          const sameHashAlgorithm = (cachedManifest?.hashAlgorithm ?? null) === (remoteManifest?.hashAlgorithm ?? null);
          if (sourceIntegrityValid && sameShards && sameHashAlgorithm) {
            await openModelStore(modelId);
            await saveManifest(remoteManifestText);
            log.info(MODULE, `Cache manifest refreshed: "${modelId}" (shards unchanged)`);
            onProgress?.({ stage: 'cache-refresh', modelId, message: `Manifest refreshed: ${modelId} (shards unchanged)`, percent: 100 });
            return {
              cached: true,
              fromCache: false,
              cacheState: 'manifest-refresh',
              modelId,
              error: null,
            };
          }

          // Cache is stale — newer model version. Delete old data before re-importing.
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
