/**
 * Kernel Tuner - Main Class
 *
 * Orchestrates kernel auto-tuning by running benchmarks with various
 * workgroup configurations and caching optimal results.
 */

import type {
  CacheKey,
  TuneRecord,
  TuneResult,
  TuneConfig,
  InputSizes,
} from './types.js';

/**
 * Kernel Tuner class
 *
 * Automatically finds optimal workgroup sizes for different kernels
 * by running benchmarks with various configurations.
 * Results are cached in localStorage for persistence across sessions.
 */
export declare class KernelTuner {
  /**
   * Initialize the tuner
   */
  init(): Promise<void>;

  /**
   * Tune a kernel by running benchmarks
   * @param kernelName - Name of kernel to tune
   * @param inputSizes - Input dimensions for tuning
   * @param options - Tuning options
   * @returns Promise resolving to tuning result
   */
  tuneKernel(
    kernelName: string,
    inputSizes: InputSizes,
    options?: TuneConfig
  ): Promise<TuneResult>;

  /**
   * Get cached tuning result
   * @param kernelName - Kernel name
   * @param inputSizes - Input sizes
   * @returns Cached result or null
   */
  getCachedResult(kernelName: string, inputSizes: InputSizes): TuneResult | null;

  /**
   * Clear all cached results
   */
  clearCache(): void;

  /**
   * Get all cached results
   * @returns Object with all cached results
   */
  getAllCachedResults(): Record<string, TuneRecord>;

  /**
   * Destroy tuner resources
   */
  destroy(): void;
}

/**
 * Get the global kernel tuner
 * @returns Promise resolving to kernel tuner instance
 */
export function getKernelTuner(): Promise<KernelTuner>;

/**
 * Convenience function to tune a kernel
 * @param kernelName - Kernel name
 * @param inputSizes - Input sizes
 * @returns Promise resolving to tuning result
 */
export function tuneKernel(
  kernelName: string,
  inputSizes: InputSizes
): Promise<TuneResult>;
