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

import { downloadModel, type DownloadProgress, type DownloadOptions } from './downloader.js';
import {
  runPreflightChecks,
  type PreflightResult,
  type ModelRequirements,
  GEMMA_1B_REQUIREMENTS,
} from './preflight.js';
import { formatBytes } from './quota.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Remote model configuration
 */
export interface RemoteModelConfig {
  /** Model identifier */
  modelId: string;
  /** Display name for UI */
  displayName: string;
  /** Base URL for shards (any static CDN) */
  baseUrl: string;
  /** Model requirements for pre-flight checks */
  requirements: ModelRequirements;
}

/**
 * Quick-start download options
 */
export interface QuickStartDownloadOptions {
  /** Progress callback */
  onProgress?: (progress: DownloadProgress) => void;
  /** Called when preflight checks complete */
  onPreflightComplete?: (result: PreflightResult) => void;
  /** Called to request storage consent from user. Return true to proceed. */
  onStorageConsent?: (requiredBytes: number, availableBytes: number, modelName: string) => Promise<boolean>;
  /** Abort signal */
  signal?: AbortSignal;
  /** Number of concurrent downloads (default: 3) */
  concurrency?: number;
  /** Skip preflight checks */
  skipPreflight?: boolean;
}

/**
 * Quick-start download result
 */
export interface QuickStartDownloadResult {
  /** Download succeeded */
  success: boolean;
  /** Model ID that was downloaded */
  modelId: string;
  /** Error message if failed */
  error?: string;
  /** Preflight result (if checks were run) */
  preflight?: PreflightResult;
  /** Was blocked by preflight */
  blockedByPreflight?: boolean;
  /** User declined consent */
  userDeclined?: boolean;
}

// ============================================================================
// Model Registry
// ============================================================================

/**
 * CDN base URL for model hosting
 * Configure this based on your hosting setup.
 * Default uses same-origin /doppler/models/ path (for Firebase Hosting or local dev)
 */
let CDN_BASE_URL = '';

/**
 * Get the auto-detected or configured CDN base URL
 */
function getEffectiveCDNBaseUrl(): string {
  if (CDN_BASE_URL) return CDN_BASE_URL;

  // Auto-detect: use same origin for Firebase Hosting or local dev
  if (typeof window !== 'undefined') {
    return `${window.location.origin}/doppler/models`;
  }
  // Fallback for Node.js/SSR
  return '/doppler/models';
}

/**
 * Set the CDN base URL for model downloads
 */
export function setCDNBaseUrl(url: string): void {
  CDN_BASE_URL = url.replace(/\/$/, ''); // Remove trailing slash
}

/**
 * Get the current CDN base URL
 */
export function getCDNBaseUrl(): string {
  return getEffectiveCDNBaseUrl();
}

/**
 * Available quick-start models
 * These are models with pre-configured requirements and hosted shards
 */
export const QUICKSTART_MODELS: Record<string, RemoteModelConfig> = {
  'gemma-3-1b-it-q4': {
    modelId: 'gemma-3-1b-it-q4',
    displayName: 'Gemma 3 1B IT (Q4)',
    baseUrl: 'https://huggingface.co/clocksmith/gemma3-1b-rdrr/resolve/main',
    requirements: GEMMA_1B_REQUIREMENTS,
  },
};

/**
 * Get quick-start model config by ID
 */
export function getQuickStartModel(modelId: string): RemoteModelConfig | undefined {
  return QUICKSTART_MODELS[modelId];
}

/**
 * List all available quick-start models
 */
export function listQuickStartModels(): RemoteModelConfig[] {
  return Object.values(QUICKSTART_MODELS);
}

/**
 * Register a custom quick-start model
 */
export function registerQuickStartModel(config: RemoteModelConfig): void {
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
 * @param modelId - Model ID (e.g., 'gemma-1b-instruct')
 * @param options - Download options
 * @returns Download result
 *
 * @example
 * ```typescript
 * const result = await downloadQuickStartModel('gemma-1b-instruct', {
 *   onProgress: (p) => updateProgressBar(p.percent),
 *   onStorageConsent: async (required, available) => {
 *     return confirm(`Download ${formatBytes(required)}?`);
 *   },
 * });
 *
 * if (result.success) {
 *   console.log('Model ready!');
 * } else if (result.blockedByPreflight) {
 *   console.error('Blocked:', result.preflight?.blockers);
 * }
 * ```
 */
export async function downloadQuickStartModel(
  modelId: string,
  options: QuickStartDownloadOptions = {}
): Promise<QuickStartDownloadResult> {
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
  let preflight: PreflightResult | undefined;

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
        error: `Preflight check failed: ${(err as Error).message}`,
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
        error: `Consent flow failed: ${(err as Error).message}`,
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

    const downloadOpts: DownloadOptions = {
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
    const errorMessage = (err as Error).message;

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
 * @param modelId - Model ID
 * @returns True if model exists in OPFS
 */
export async function isModelDownloaded(modelId: string): Promise<boolean> {
  // Import dynamically to avoid circular deps
  const { modelExists } = await import('./shard-manager.js');
  return modelExists(modelId);
}

/**
 * Get download size for a quick-start model
 *
 * @param modelId - Model ID
 * @returns Size in bytes, or null if unknown model
 */
export function getModelDownloadSize(modelId: string): number | null {
  const config = QUICKSTART_MODELS[modelId];
  return config?.requirements.downloadSize ?? null;
}

/**
 * Format model info for display
 */
export function formatModelInfo(modelId: string): string | null {
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
