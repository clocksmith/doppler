

import { getRuntimeConfig } from '../config/runtime.js';

// Constants (IndexedDB)
export const DB_NAME = 'doppler-download-state';
export const DB_VERSION = 1;
export const STORE_NAME = 'downloads';


export function getDistributionConfig() {
  return getRuntimeConfig().loading.distribution;
}


export function getDefaultConcurrency() {
  return getDistributionConfig().concurrentDownloads;
}


export function getMaxRetries() {
  return getDistributionConfig().maxRetries;
}


export function getInitialRetryDelayMs() {
  return getDistributionConfig().initialRetryDelayMs;
}


export function getMaxRetryDelayMs() {
  return getDistributionConfig().maxRetryDelayMs;
}


export function getCdnBasePath() {
  return getDistributionConfig().cdnBasePath;
}


export function getProgressUpdateIntervalMs() {
  return getDistributionConfig().progressUpdateIntervalMs;
}


export function getRequiredContentEncoding() {
  return getDistributionConfig().requiredContentEncoding;
}
