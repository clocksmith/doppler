import { modelExists } from '../storage/shard-manager.js';
import { downloadModel } from '../storage/downloader.js';
import { isOPFSAvailable } from '../storage/quota.js';
import { log } from '../debug/index.js';

const MODULE = 'OPFSCache';

function toErrorMessage(error) {
  if (error instanceof Error && typeof error.message === 'string' && error.message.length > 0) {
    return error.message;
  }
  return String(error);
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
      log.info(MODULE, `Cache hit: "${modelId}"`);
      return { cached: true, fromCache: true, modelId, error: null };
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
