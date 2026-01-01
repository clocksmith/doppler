/**
 * Download Types
 *
 * Type definitions for the resumable model downloader.
 *
 * @module storage/download-types
 */

import type { RDRRManifest } from './rdrr-format.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../config/index.js';

/**
 * Download progress information
 */
export interface DownloadProgress {
  modelId: string;
  manifest?: RDRRManifest;
  totalShards: number;
  completedShards: number;
  totalBytes: number;
  downloadedBytes: number;
  percent: number;
  status: DownloadStatus;
  currentShard: number | null;
  speed: number;
  stage?: string;
}

/**
 * Shard download progress
 */
export interface ShardProgress {
  shardIndex: number;
  receivedBytes: number;
  totalBytes: number;
  percent: number;
}

/**
 * Download status values
 */
export type DownloadStatus = 'downloading' | 'paused' | 'completed' | 'error';

/**
 * Persisted download state
 */
export interface DownloadState {
  modelId: string;
  baseUrl: string;
  manifest: RDRRManifest;
  completedShards: Set<number>;
  startTime: number;
  status: DownloadStatus;
  error?: string;
}

/**
 * Serializable download state for IndexedDB
 */
export interface SerializedDownloadState {
  modelId: string;
  baseUrl: string;
  manifest: RDRRManifest;
  completedShards: number[];
  startTime: number;
  status: DownloadStatus;
  error?: string;
}

/**
 * Download options
 */
export interface DownloadOptions {
  concurrency?: number;
  requestPersist?: boolean;
  modelId?: string;
  signal?: AbortSignal;
}

/**
 * Retry policy configuration
 */
export interface RetryPolicy {
  maxRetries: number;
  backoffMs: number;
  backoffMultiplier: number;
}

/**
 * Download need check result
 */
export interface DownloadNeededResult {
  needed: boolean;
  reason: string;
  missingShards: number[];
}

/**
 * Active download tracking
 */
export interface ActiveDownload {
  state: DownloadState;
  abortController: AbortController;
}

/**
 * Speed tracking helper
 */
export interface SpeedTracker {
  lastBytes: number;
  lastTime: number;
  speed: number;
}

/**
 * Progress callback type
 */
export type ProgressCallback = (progress: DownloadProgress) => void;

// Constants (IndexedDB)
export const DB_NAME = 'doppler-download-state';
export const DB_VERSION = 1;
export const STORE_NAME = 'downloads';

// Re-export from config for backwards compatibility
export const DEFAULT_CONCURRENCY = DEFAULT_DISTRIBUTION_CONFIG.concurrentDownloads;
export const MAX_RETRIES = DEFAULT_DISTRIBUTION_CONFIG.maxRetries;
export const INITIAL_RETRY_DELAY = DEFAULT_DISTRIBUTION_CONFIG.initialRetryDelayMs;
export const MAX_RETRY_DELAY = DEFAULT_DISTRIBUTION_CONFIG.maxRetryDelayMs;
