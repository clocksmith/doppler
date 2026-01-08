/**
 * quickstart-downloader.ts - Quick-Start Model Downloader
 *
 * Provides a streamlined API for the quick-start download flow:
 * - Pre-flight checks (VRAM, storage, GPU)
 * - User consent flow
 * - Parallel shard fetching with progress
 *
 * Works with any static file CDN (Firebase Hosting, S3, Cloudflare, etc.)
 *
 * @module storage/quickstart-downloader
 */

import { downloadModel } from './downloader.js';
import {
  runPreflightChecks,
  GEMMA_1B_REQUIREMENTS,
} from './preflight.js';
import { formatBytes } from './quota.js';
import { getCdnBasePath } from './download-types.js';

// ============================================================================
// Model Registry
// ============================================================================

/**
 * CDN base URL for model hosting
 * Configure this based on your hosting setup.
 * Default uses config value (null = same-origin /doppler/models/ path for Firebase Hosting or local dev)
 * @type {string | null}
 */
let cdnBaseOverride = null;

/**
 * Get the auto-detected or configured CDN base URL
 * @returns {string}
 */
function getEffectiveCDNBaseUrl() {
  const runtimeBase = getCdnBasePath();
  const base = cdnBaseOverride ?? runtimeBase ?? '';
  if (base) return base;

  // Auto-detect: use same origin for Firebase Hosting or local dev
  if (typeof window !== 'undefined') {
    return `${window.location.origin}/doppler/models`;
  }
  // Fallback for Node.js/SSR
  return '/doppler/models';
}

/**
 * Set the CDN base URL for model downloads
 * @param {string} url
 * @returns {void}
 */
export function setCDNBaseUrl(url) {
  cdnBaseOverride = url.replace(/\/$/, ''); // Remove trailing slash
}

/**
 * Get the current CDN base URL
 * @returns {string}
 */
export function getCDNBaseUrl() {
  return getEffectiveCDNBaseUrl();
}

/**
 * Available quick-start models
 * These are models with pre-configured requirements and hosted shards
 * @type {Record<string, import('./quickstart-downloader.js').RemoteModelConfig>}
 */
export const QUICKSTART_MODELS = {
  'gemma-3-1b-it-q4': {
    modelId: 'gemma-3-1b-it-q4',
    displayName: 'Gemma 3 1B IT (Q4)',
    baseUrl: 'https://huggingface.co/clocksmith/gemma3-1b-rdrr/resolve/main',
    requirements: GEMMA_1B_REQUIREMENTS,
  },
};

/**
 * Get quick-start model config by ID
 * @param {string} modelId
 * @returns {import('./quickstart-downloader.js').RemoteModelConfig | undefined}
 */
export function getQuickStartModel(modelId) {
  return QUICKSTART_MODELS[modelId];
}

/**
 * List all available quick-start models
 * @returns {import('./quickstart-downloader.js').RemoteModelConfig[]}
 */
export function listQuickStartModels() {
  return Object.values(QUICKSTART_MODELS);
}

/**
 * Register a custom quick-start model
 * @param {import('./quickstart-downloader.js').RemoteModelConfig} config
 * @returns {void}
 */
export function registerQuickStartModel(config) {
  QUICKSTART_MODELS[config.modelId] = config;
}

// ============================================================================
// Download Functions
// ============================================================================

/**
 * Download a quick-start model
 *
 * Flow:
 * 1. Run pre-flight checks (VRAM, storage, GPU)
 * 2. If checks fail, return early with blockers
 * 3. Request user consent for storage usage
 * 4. If declined, return early
 * 5. Download model with progress updates
 *
 * @param {string} modelId - Model ID (e.g., 'gemma-1b-instruct')
 * @param {import('./quickstart-downloader.js').QuickStartDownloadOptions} [options] - Download options
 * @returns {Promise<import('./quickstart-downloader.js').QuickStartDownloadResult>} Download result
 *
 * @example
 * ```typescript
 * import { log } from '../debug/index.js';
 *
 * const result = await downloadQuickStartModel('gemma-1b-instruct', {
 *   onProgress: (p) => updateProgressBar(p.percent),
 *   onStorageConsent: async (required, available) => {
 *     return confirm(`Download ${formatBytes(required)}?`);
 *   },
 * });
 *
 * if (result.success) {
 *   log.info('Quickstart', 'Model ready!');
 * } else if (result.blockedByPreflight) {
 *   log.warn('Quickstart', 'Blocked by preflight', result.preflight?.blockers);
 * }
 * ```
 */
export async function downloadQuickStartModel(
  modelId,
  options = {}
) {
  const config = QUICKSTART_MODELS[modelId];

  if (!config) {
    return {
      success: false,
      modelId,
      error: `Unknown model: ${modelId}. Available: ${Object.keys(QUICKSTART_MODELS).join(', ')}`,
    };
  }

  const {
    onProgress,
    onPreflightComplete,
    onStorageConsent,
    signal,
    concurrency = 3,
    skipPreflight = false,
  } = options;

  // -------------------------------------------------------------------------
  // Step 1: Pre-flight checks
  // -------------------------------------------------------------------------
  /** @type {import('./preflight.js').PreflightResult | undefined} */
  let preflight;

  if (!skipPreflight) {
    try {
      preflight = await runPreflightChecks(config.requirements);
      onPreflightComplete?.(preflight);

      if (!preflight.canProceed) {
        return {
          success: false,
          modelId,
          error: preflight.blockers.join('; '),
          preflight,
          blockedByPreflight: true,
        };
      }
    } catch (err) {
      return {
        success: false,
        modelId,
        error: `Preflight check failed: ${/** @type {Error} */ (err).message}`,
      };
    }
  }

  // -------------------------------------------------------------------------
  // Step 2: Request user consent
  // -------------------------------------------------------------------------
  if (onStorageConsent) {
    const requiredBytes = config.requirements.downloadSize;
    const availableBytes = preflight?.storage.available ?? 0;

    try {
      const consent = await onStorageConsent(requiredBytes, availableBytes, config.displayName);

      if (!consent) {
        return {
          success: false,
          modelId,
          error: 'User declined storage consent',
          preflight,
          userDeclined: true,
        };
      }
    } catch (err) {
      return {
        success: false,
        modelId,
        error: `Consent flow failed: ${/** @type {Error} */ (err).message}`,
        preflight,
      };
    }
  }

  // -------------------------------------------------------------------------
  // Step 3: Download model
  // -------------------------------------------------------------------------
  try {
    // Check for abort before starting
    if (signal?.aborted) {
      return {
        success: false,
        modelId,
        error: 'Download aborted',
        preflight,
      };
    }

    /** @type {import('./download-types.js').DownloadOptions} */
    const downloadOpts = {
      concurrency,
      requestPersist: true,
      modelId: config.modelId,
      signal,
    };

    const success = await downloadModel(
      config.baseUrl,
      onProgress,
      downloadOpts
    );

    if (!success) {
      return {
        success: false,
        modelId,
        error: 'Download failed',
        preflight,
      };
    }

    return {
      success: true,
      modelId,
      preflight,
    };
  } catch (err) {
    const errorMessage = /** @type {Error} */ (err).message;

    // Handle specific error types
    if (errorMessage.includes('aborted') || signal?.aborted) {
      return {
        success: false,
        modelId,
        error: 'Download aborted by user',
        preflight,
      };
    }

    if (errorMessage.includes('quota') || errorMessage.includes('storage')) {
      return {
        success: false,
        modelId,
        error: `Storage error: ${errorMessage}`,
        preflight,
      };
    }

    return {
      success: false,
      modelId,
      error: `Download failed: ${errorMessage}`,
      preflight,
    };
  }
}

/**
 * Check if a quick-start model is already downloaded
 *
 * @param {string} modelId - Model ID
 * @returns {Promise<boolean>} True if model exists in OPFS
 */
export async function isModelDownloaded(modelId) {
  // Import dynamically to avoid circular deps
  const { modelExists } = await import('./shard-manager.js');
  return modelExists(modelId);
}

/**
 * Get download size for a quick-start model
 *
 * @param {string} modelId - Model ID
 * @returns {number | null} Size in bytes, or null if unknown model
 */
export function getModelDownloadSize(modelId) {
  const config = QUICKSTART_MODELS[modelId];
  return config?.requirements.downloadSize ?? null;
}

/**
 * Format model info for display
 * @param {string} modelId
 * @returns {string | null}
 */
export function formatModelInfo(modelId) {
  const config = QUICKSTART_MODELS[modelId];
  if (!config) return null;

  const { requirements } = config;
  return [
    config.displayName,
    `${requirements.paramCount} parameters`,
    `${requirements.quantization} quantization`,
    `${formatBytes(requirements.downloadSize)} download`,
    `${formatBytes(requirements.vramRequired)} VRAM required`,
  ].join(' | ');
}
