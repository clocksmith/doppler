

import { log } from '../src/debug/index.js';
import {
  downloadQuickStartModel,
  QUICKSTART_MODELS,
} from '../src/storage/quickstart-downloader.js';

/**
 * Controls the quick-start model download flow.
 */
export class QuickStartController {
  /** @type {QuickStartCallbacks} */
  #callbacks;

  /**
   * @param {QuickStartCallbacks} callbacks
   */
  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  /**
   * Get available quick-start models.
   * @returns {Record<string, QuickStartConfig>}
   */
  static getAvailableModels() {
    return QUICKSTART_MODELS;
  }

  /**
   * Check if a model ID is a quick-start model.
   * @param {string} modelId
   * @returns {boolean}
   */
  static isQuickStartModel(modelId) {
    return modelId in QUICKSTART_MODELS;
  }

  /**
   * Start the quick-start download flow for a model.
   * @param {string} modelId
   * @returns {Promise<QuickStartResult>}
   */
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

/**
 * @typedef {Object} QuickStartCallbacks
 * @property {(requiredBytes: number, availableBytes: number) => void} [onVRAMInsufficient]
 * @property {(modelName: string, requiredBytes: number, availableBytes: number) => Promise<boolean>} [onStorageConsent]
 * @property {() => void} [onDownloadStart]
 * @property {(percent: number, downloadedBytes: number, totalBytes: number, speed: number) => void} [onProgress]
 * @property {(modelId: string) => void} [onComplete]
 * @property {() => void} [onDeclined]
 * @property {(error: string) => void} [onError]
 */

/**
 * @typedef {Object} QuickStartResult
 * @property {boolean} success
 * @property {string} [error]
 * @property {boolean} [blockedByPreflight]
 * @property {boolean} [userDeclined]
 */

/**
 * @typedef {Object} QuickStartConfig
 * @property {string} displayName
 * @property {string} baseUrl
 * @property {QuickStartRequirements} requirements
 */

/**
 * @typedef {Object} QuickStartRequirements
 * @property {string} architecture
 * @property {string} quantization
 * @property {number} downloadSize
 * @property {string} paramCount
 */
