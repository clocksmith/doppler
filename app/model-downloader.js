

import { log } from '../src/debug/index.js';
import { downloadModel } from '../src/storage/downloader.js';
import { deleteModel as deleteModelFromStorage } from '../src/storage/shard-manager.js';

export class ModelDownloader {
  #callbacks;

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  async download(model, options = {}) {
    const sources = model.sources || {};

    // Determine URL: prefer server, then remote
    let downloadUrl = null;
    let storageId = model.key.replace(/[^a-zA-Z0-9_-]/g, '_');

    if (sources.server) {
      downloadUrl = sources.server.url;
    } else if (sources.remote) {
      downloadUrl = sources.remote.url;
      storageId = sources.remote.id || storageId;
    }

    if (!downloadUrl) {
      throw new Error('No download source available');
    }

    log.info('ModelDownloader', `Downloading "${model.name}" from: ${downloadUrl}`);

    try {
      const success = await downloadModel(
      downloadUrl,
      (progress) => {
        const percent = progress.totalBytes > 0
        ? Math.round((progress.downloadedBytes / progress.totalBytes) * 100)
        : 0;

        this.#callbacks.onProgress?.(model.key, percent, progress);

        if (progress.stage === 'verifying') {
          this.#callbacks.onStatus?.('verifying');
        }
      },
      { modelId: storageId }
      );

      if (!success) {
        throw new Error('Download failed');
      }

      log.info('ModelDownloader', `Download complete: ${model.name}`);
      return true;
    } catch (error) {
      log.error('ModelDownloader', 'Download failed:', error);
      this.#callbacks.onProgress?.(model.key, 0, null);
      throw error;
    }
  }

  async delete(model) {
    const sources = model.sources || {};
    const browserId = sources.browser?.id;

    if (!browserId) {
      throw new Error('Model is not cached in browser');
    }

    log.info('ModelDownloader', `Deleting cached model: ${model.name} (${browserId})`);

    await deleteModelFromStorage(browserId);
    log.info('ModelDownloader', `Deleted: ${model.name}`);
  }
}




