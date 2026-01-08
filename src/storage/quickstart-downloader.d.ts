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

import type { DownloadProgress } from './downloader.js';
import type { PreflightResult, ModelRequirements } from './preflight.js';

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

/**
 * Set the CDN base URL for model downloads
 */
export declare function setCDNBaseUrl(url: string): void;

/**
 * Get the current CDN base URL
 */
export declare function getCDNBaseUrl(): string;

/**
 * Available quick-start models
 * These are models with pre-configured requirements and hosted shards
 */
export declare const QUICKSTART_MODELS: Record<string, RemoteModelConfig>;

/**
 * Get quick-start model config by ID
 */
export declare function getQuickStartModel(modelId: string): RemoteModelConfig | undefined;

/**
 * List all available quick-start models
 */
export declare function listQuickStartModels(): RemoteModelConfig[];

/**
 * Register a custom quick-start model
 */
export declare function registerQuickStartModel(config: RemoteModelConfig): void;

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
export declare function downloadQuickStartModel(
  modelId: string,
  options?: QuickStartDownloadOptions
): Promise<QuickStartDownloadResult>;

/**
 * Check if a quick-start model is already downloaded
 *
 * @param modelId - Model ID
 * @returns True if model exists in OPFS
 */
export declare function isModelDownloaded(modelId: string): Promise<boolean>;

/**
 * Get download size for a quick-start model
 *
 * @param modelId - Model ID
 * @returns Size in bytes, or null if unknown model
 */
export declare function getModelDownloadSize(modelId: string): number | null;

/**
 * Format model info for display
 */
export declare function formatModelInfo(modelId: string): string | null;
