import {
  modelExists,
  openModelStore,
  loadManifestFromStore,
  saveManifest,
} from '../storage/shard-manager.js';
import { downloadModel } from '../storage/downloader.js';
import { isOPFSAvailable } from '../storage/quota.js';
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
      defaultKernelPath: inference.defaultKernelPath ?? null,
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

export async function ensureModelCached(modelId, modelBaseUrl) {
  if (!modelId || !modelBaseUrl) {
    return { cached: false, fromCache: false, modelId, error: 'missing-args' };
  }

  if (!isOPFSAvailable()) {
    log.warn(MODULE, 'OPFS not available in this browser');
    return { cached: false, fromCache: false, modelId, error: 'opfs-unavailable' };
  }

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
          if (sourceIntegrityValid && cachedFingerprint === remoteFingerprint) {
            log.info(MODULE, `Cache hit: "${modelId}"`);
            return { cached: true, fromCache: true, modelId, error: null };
          }

          const sameShards = hasSameShardSet(cachedManifest, remoteManifest);
          const sameHashAlgorithm = (cachedManifest?.hashAlgorithm ?? null) === (remoteManifest?.hashAlgorithm ?? null);
          if (sourceIntegrityValid && sameShards && sameHashAlgorithm) {
            await openModelStore(modelId);
            await saveManifest(remoteManifestText);
            log.info(MODULE, `Cache manifest refreshed: "${modelId}" (shards unchanged)`);
            return { cached: true, fromCache: false, modelId, error: null };
          }
          log.info(MODULE, `Cache stale: "${modelId}" manifest/shards changed; re-importing`);
        }
      } catch (error) {
        const message = toErrorMessage(error);
        log.warn(MODULE, `Cache validation failed (${message}); refusing cached model "${modelId}"`);
        return { cached: false, fromCache: false, modelId, error: message };
      }
    }
  } catch (error) {
    const message = toErrorMessage(error);
    log.warn(MODULE, `Cache check failed: ${message}`);
    return { cached: false, fromCache: false, modelId, error: message };
  }

  log.info(MODULE, `Cache miss: "${modelId}". Importing from ${modelBaseUrl}...`);

  try {
    const success = await downloadModel(modelBaseUrl, (progress) => {
      if (!progress) return;
      const shard = Number.isFinite(progress.completedShards) ? progress.completedShards : '?';
      const total = Number.isFinite(progress.totalShards) ? progress.totalShards : '?';
      const mb = Number.isFinite(progress.downloadedBytes)
        ? (progress.downloadedBytes / (1024 * 1024)).toFixed(1)
        : '?';
      log.verbose(MODULE, `Shard ${shard}/${total} (${mb} MB)`);
    });

    if (success) {
      log.info(MODULE, `Import complete: "${modelId}"`);
      return { cached: true, fromCache: false, modelId, error: null };
    }
    return { cached: false, fromCache: false, modelId, error: 'download-incomplete' };
  } catch (error) {
    const message = toErrorMessage(error);
    log.error(MODULE, `Import failed: ${message}`);
    return { cached: false, fromCache: false, modelId, error: message };
  }
}
