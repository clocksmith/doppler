/**
 * Download Types
 *
 * Type definitions for the resumable model downloader.
 *
 * @module storage/download-types
 */

import { getRuntimeConfig } from '../config/runtime.js';

// Constants (IndexedDB)
export const DB_NAME = 'doppler-download-state';
export const DB_VERSION = 1;
export const STORE_NAME = 'downloads';

/**
 * @returns {import('../config/schema/runtime-schema.js').DistributionConfig}
 */
export function getDistributionConfig() {
  return getRuntimeConfig().loading.distribution;
}

/**
 * @returns {number}
 */
export function getDefaultConcurrency() {
  return getDistributionConfig().concurrentDownloads;
}

/**
 * @returns {number}
 */
export function getMaxRetries() {
  return getDistributionConfig().maxRetries;
}

/**
 * @returns {number}
 */
export function getInitialRetryDelayMs() {
  return getDistributionConfig().initialRetryDelayMs;
}

/**
 * @returns {number}
 */
export function getMaxRetryDelayMs() {
  return getDistributionConfig().maxRetryDelayMs;
}

/**
 * @returns {string | null}
 */
export function getCdnBasePath() {
  return getDistributionConfig().cdnBasePath;
}

/**
 * @returns {number}
 */
export function getProgressUpdateIntervalMs() {
  return getDistributionConfig().progressUpdateIntervalMs;
}
