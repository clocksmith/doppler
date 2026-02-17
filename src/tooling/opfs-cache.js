import { modelExists } from '../storage/shard-manager.js';
import { downloadModel } from '../storage/downloader.js';
import { isOPFSAvailable } from '../storage/quota.js';

/**
 * Ensures a model is cached in OPFS. Downloads from the given base URL on
 * cache miss. Returns a result object — never throws (caller falls back to
 * the HTTP shard-loader path on any error).
 *
 * @param {string} modelId
 * @param {string} modelBaseUrl  Base URL served by the static file server
 *                               (e.g. '/models/local/gemma-3-1b-it-wf16')
 * @returns {Promise<{cached: boolean, fromCache: boolean, modelId: string, error: string|null}>}
 */
export async function ensureModelCached(modelId, modelBaseUrl) {
  if (!modelId || !modelBaseUrl) {
    return { cached: false, fromCache: false, modelId, error: 'missing-args' };
  }

  if (!isOPFSAvailable()) {
    console.warn('[opfs-cache] OPFS not available in this browser');
    return { cached: false, fromCache: false, modelId, error: 'opfs-unavailable' };
  }

  try {
    const exists = await modelExists(modelId);
    if (exists) {
      console.log(`[opfs-cache] Cache hit: "${modelId}"`);
      return { cached: true, fromCache: true, modelId, error: null };
    }
  } catch (error) {
    console.warn(`[opfs-cache] Cache check failed: ${error.message}`);
    return { cached: false, fromCache: false, modelId, error: error.message };
  }

  console.log(`[opfs-cache] Cache miss: "${modelId}". Importing from ${modelBaseUrl}...`);

  try {
    const success = await downloadModel(modelBaseUrl, (progress) => {
      if (!progress) return;
      const shard = Number.isFinite(progress.completedShards) ? progress.completedShards : '?';
      const total = Number.isFinite(progress.totalShards) ? progress.totalShards : '?';
      const mb = Number.isFinite(progress.downloadedBytes)
        ? (progress.downloadedBytes / (1024 * 1024)).toFixed(1)
        : '?';
      console.log(`[opfs-cache] shard ${shard}/${total} (${mb} MB)`);
    });

    if (success) {
      console.log(`[opfs-cache] Import complete: "${modelId}"`);
      return { cached: true, fromCache: false, modelId, error: null };
    }
    return { cached: false, fromCache: false, modelId, error: 'download-incomplete' };
  } catch (error) {
    console.error(`[opfs-cache] Import failed: ${error.message}`);
    return { cached: false, fromCache: false, modelId, error: error.message };
  }
}
