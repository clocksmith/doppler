/**
 * Kernel Tuner Cache
 *
 * LocalStorage caching logic for kernel tuning results.
 * Persists optimal workgroup configurations across browser sessions.
 */

import type { CacheKey, TuneRecord, KernelCapabilities, InputSizes } from './types.js';

/**
 * Get tuner configuration from runtime config
 */
export function getTunerConfig(): {
  cacheKeyPrefix: string;
  warmupIterations: number;
  timedIterations: number;
};

/**
 * Generate device signature for cache key
 * @param capabilities - Kernel capabilities containing adapter info
 * @returns Device signature string
 */
export function getDeviceSignature(capabilities: KernelCapabilities | null): string;

/**
 * Generate cache key for a kernel and input sizes
 * @param kernelName - Name of the kernel
 * @param inputSizes - Input dimensions
 * @returns Cache key string
 */
export function generateCacheKey(kernelName: string, inputSizes: InputSizes): CacheKey;

/**
 * Load cached tuning results from localStorage
 * @param capabilities - Kernel capabilities for device signature
 * @returns Map of cached tuning records
 */
export function loadCache(capabilities: KernelCapabilities | null): Map<CacheKey, TuneRecord>;

/**
 * Save cached results to localStorage
 * @param cache - Map of tuning records to save
 * @param capabilities - Kernel capabilities for device signature
 */
export function saveCache(
  cache: Map<CacheKey, TuneRecord>,
  capabilities: KernelCapabilities | null
): void;

/**
 * Clear cache from localStorage
 * @param capabilities - Kernel capabilities for device signature
 */
export function clearCacheStorage(capabilities: KernelCapabilities | null): void;
