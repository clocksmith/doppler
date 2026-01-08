/**
 * downloader.ts - Resumable Model Downloader
 *
 * Handles:
 * - Chunked downloads with progress reporting
 * - Resume support via IndexedDB state tracking
 * - Parallel shard downloads with concurrency control
 * - Automatic retry with exponential backoff
 * - Quota checking before downloads
 *
 * @module storage/downloader
 */

// Re-export types from download-types
export type {
  DownloadProgress,
  ShardProgress,
  DownloadStatus,
  DownloadState,
  DownloadOptions,
  RetryPolicy,
  DownloadNeededResult,
  ProgressCallback,
} from './download-types.js';

import type {
  DownloadProgress,
  DownloadOptions,
  DownloadNeededResult,
  ProgressCallback,
} from './download-types.js';

/**
 * Downloads a model with progress reporting and resume support
 */
export declare function downloadModel(
  baseUrl: string,
  onProgress?: ProgressCallback,
  options?: DownloadOptions
): Promise<boolean>;

/**
 * Pauses an active download
 */
export declare function pauseDownload(modelId: string): boolean;

/**
 * Resumes a paused download
 */
export declare function resumeDownload(
  modelId: string,
  onProgress?: ProgressCallback,
  options?: DownloadOptions
): Promise<boolean>;

/**
 * Gets the download progress for a model
 */
export declare function getDownloadProgress(modelId: string): Promise<DownloadProgress | null>;

/**
 * Lists all in-progress or paused downloads
 */
export declare function listDownloads(): Promise<DownloadProgress[]>;

/**
 * Cancels and removes a download
 */
export declare function cancelDownload(modelId: string): Promise<boolean>;

/**
 * Checks if a model needs downloading
 */
export declare function checkDownloadNeeded(modelId: string): Promise<DownloadNeededResult>;

/**
 * Formats download speed for display
 */
export declare function formatSpeed(bytesPerSecond: number): string;

/**
 * Estimates remaining download time
 */
export declare function estimateTimeRemaining(remainingBytes: number, bytesPerSecond: number): string;
