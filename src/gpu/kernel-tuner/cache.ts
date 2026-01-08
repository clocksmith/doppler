/**
 * Kernel Tuner Cache
 *
 * LocalStorage caching logic for kernel tuning results.
 * Persists optimal workgroup configurations across browser sessions.
 */

import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import type { CacheKey, TuneRecord, DeviceInfo, KernelCapabilities, InputSizes } from './types.js';

/**
 * Get tuner configuration from runtime config
 */
export function getTunerConfig() {
  return getRuntimeConfig().tuner;
}

/**
 * Generate device signature for cache key
 * @param capabilities - Kernel capabilities containing adapter info
 * @returns Device signature string
 */
export function getDeviceSignature(capabilities: KernelCapabilities | null): string {
  const info: DeviceInfo = capabilities?.adapterInfo || { vendor: '', architecture: '', device: '' };
  return `${info.vendor}_${info.architecture}_${info.device}`.replace(/[^a-zA-Z0-9]/g, '_');
}

/**
 * Generate cache key for a kernel and input sizes
 * @param kernelName - Name of the kernel
 * @param inputSizes - Input dimensions
 * @returns Cache key string
 */
export function generateCacheKey(kernelName: string, inputSizes: InputSizes): CacheKey {
  return `${kernelName}_${JSON.stringify(inputSizes)}`;
}

/**
 * Load cached tuning results from localStorage
 * @param capabilities - Kernel capabilities for device signature
 * @returns Map of cached tuning records
 */
export function loadCache(capabilities: KernelCapabilities | null): Map<CacheKey, TuneRecord> {
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
 * @param cache - Map of tuning records to save
 * @param capabilities - Kernel capabilities for device signature
 */
export function saveCache(
  cache: Map<CacheKey, TuneRecord>,
  capabilities: KernelCapabilities | null
): void {
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
 * @param capabilities - Kernel capabilities for device signature
 */
export function clearCacheStorage(capabilities: KernelCapabilities | null): void {
  if (typeof localStorage === 'undefined') return;

  const signature = getDeviceSignature(capabilities);
  localStorage.removeItem(getTunerConfig().cacheKeyPrefix + signature);
}
