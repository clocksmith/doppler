

import { log } from '../src/debug/index.js';
import {
  downloadQuickStartModel,
  QUICKSTART_MODELS,
} from '../src/storage/quickstart-downloader.js';

export class QuickStartController {
  #callbacks;

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  static getAvailableModels() {
    return QUICKSTART_MODELS;
  }

  static isQuickStartModel(modelId) {
    return modelId in QUICKSTART_MODELS;
  }

  async start(modelId) {
    const config = QUICKSTART_MODELS[modelId];
    if (!config) {
      return {
        success: false,
        error: `Unknown quick-start model: ${modelId}`,
      };
    }

    log.info('QuickStart', `Starting download for ${modelId}`);

    const result = await downloadQuickStartModel(modelId, {
      onPreflightComplete: (preflight) => {
        log.debug('QuickStart', 'Preflight:', preflight);

        if (!preflight.vram.sufficient) {
          this.#callbacks.onVRAMInsufficient?.(
          preflight.vram.required,
          preflight.vram.available
          );
        }
      },
      onStorageConsent: async (required, available, modelName) => {
        const consent = await this.#callbacks.onStorageConsent?.(
        modelName,
        required,
        available
        );
        if (consent) {
          this.#callbacks.onDownloadStart?.();
        }
        return consent ?? false;
      },
      onProgress: (progress) => {
        this.#callbacks.onProgress?.(
        progress.percent,
        progress.downloadedBytes,
        progress.totalBytes,
        progress.speed
        );
      },
    });

    if (result.success) {
      log.info('QuickStart', `Download complete for ${modelId}`);
      this.#callbacks.onComplete?.(modelId);
    } else if (result.blockedByPreflight) {
      log.debug('QuickStart', 'Blocked by preflight:', result.error);
    } else if (result.userDeclined) {
      log.debug('QuickStart', 'User declined');
      this.#callbacks.onDeclined?.();
    } else {
      this.#callbacks.onError?.(result.error || 'Download failed');
    }

    return result;
  }
}




