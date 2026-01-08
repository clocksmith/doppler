/**
 * Kernel Tuner Cache
 *
 * LocalStorage caching logic for kernel tuning results.
 * Persists optimal workgroup configurations across browser sessions.
 */

import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';

/**
 * Get tuner configuration from runtime config
 * @returns {ReturnType<typeof import('./cache.js').getTunerConfig>}
 */
export function getTunerConfig() {
  return getRuntimeConfig().tuner;
}

/**
 * Generate device signature for cache key
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities containing adapter info
 * @returns {string} Device signature string
 */
export function getDeviceSignature(capabilities) {
  /** @type {import('./types.js').DeviceInfo} */
  const info = capabilities?.adapterInfo || { vendor: '', architecture: '', device: '' };
  return `${info.vendor}_${info.architecture}_${info.device}`.replace(/[^a-zA-Z0-9]/g, '_');
}

/**
 * Generate cache key for a kernel and input sizes
 * @param {string} kernelName - Name of the kernel
 * @param {import('./types.js').InputSizes} inputSizes - Input dimensions
 * @returns {import('./types.js').CacheKey} Cache key string
 */
export function generateCacheKey(kernelName, inputSizes) {
  return `${kernelName}_${JSON.stringify(inputSizes)}`;
}

/**
 * Load cached tuning results from localStorage
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities for device signature
 * @returns {Map<import('./types.js').CacheKey, import('./types.js').TuneRecord>} Map of cached tuning records
 */
export function loadCache(capabilities) {
  if (typeof localStorage === 'undefined') {
    return new Map();
  }

  const signature = getDeviceSignature(capabilities);
  const cacheKey = getTunerConfig().cacheKeyPrefix + signature;

  try {
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      const data = JSON.parse(cached);
      return new Map(Object.entries(data));
    }
  } catch (e) {
    log.warn('KernelTuner', `Failed to load cache: ${e}`);
  }

  return new Map();
}

/**
 * Save cached results to localStorage
 * @param {Map<import('./types.js').CacheKey, import('./types.js').TuneRecord>} cache - Map of tuning records to save
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities for device signature
 * @returns {void}
 */
export function saveCache(cache, capabilities) {
  if (typeof localStorage === 'undefined') return;

  const signature = getDeviceSignature(capabilities);
  const cacheKey = getTunerConfig().cacheKeyPrefix + signature;

  try {
    const data = Object.fromEntries(cache);
    localStorage.setItem(cacheKey, JSON.stringify(data));
  } catch (e) {
    log.warn('KernelTuner', `Failed to save cache: ${e}`);
  }
}

/**
 * Clear cache from localStorage
 * @param {import('./types.js').KernelCapabilities | null} capabilities - Kernel capabilities for device signature
 * @returns {void}
 */
export function clearCacheStorage(capabilities) {
  if (typeof localStorage === 'undefined') return;

  const signature = getDeviceSignature(capabilities);
  localStorage.removeItem(getTunerConfig().cacheKeyPrefix + signature);
}
