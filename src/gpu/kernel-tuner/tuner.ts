/**
 * Kernel Tuner - Main Class
 *
 * Orchestrates kernel auto-tuning by running benchmarks with various
 * workgroup configurations and caching optimal results.
 */

import { getDevice, getKernelCapabilities, getDeviceLimits } from '../device.js';
import { GPUProfiler } from '../profiler.js';
import type {
  DeviceLimits,
  KernelCapabilities,
  CacheKey,
  TuneRecord,
  TuneResult,
  TuneConfig,
  InputSizes,
  WorkgroupSize,
} from './types.js';
import {
  getTunerConfig,
  loadCache,
  saveCache,
  clearCacheStorage,
  generateCacheKey,
} from './cache.js';
import {
  tuneMatmul,
  tuneAttention,
  tuneSoftmax,
  tuneRMSNorm,
  tuneDequant,
  tuneGeneric,
} from './benchmarks.js';

/**
 * Kernel Tuner class
 *
 * Automatically finds optimal workgroup sizes for different kernels
 * by running benchmarks with various configurations.
 * Results are cached in localStorage for persistence across sessions.
 */
export class KernelTuner {
  private device: GPUDevice | null;
  private profiler: GPUProfiler | null;
  private limits: DeviceLimits | null;
  private capabilities: KernelCapabilities | null;
  private cache: Map<CacheKey, TuneRecord>;

  constructor() {
    this.device = null;
    this.profiler = null;
    this.limits = null;
    this.capabilities = null;
    this.cache = new Map();
  }

  /**
   * Initialize the tuner
   */
  async init(): Promise<void> {
    this.device = getDevice();
    if (!this.device) {
      throw new Error('GPU device not initialized');
    }

    this.profiler = new GPUProfiler(this.device);
    this.limits = getDeviceLimits();
    this.capabilities = getKernelCapabilities();

    // Load cached results
    this.cache = loadCache(this.capabilities);
  }

  /**
   * Generate workgroup size candidates based on device limits
   * @private
   */
  private _generateWorkgroupCandidates(): WorkgroupSize[] {
    const maxX = this.limits?.maxComputeWorkgroupSizeX || 256;
    const maxY = this.limits?.maxComputeWorkgroupSizeY || 256;
    const maxInvocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;

    const candidates: WorkgroupSize[] = [];

    // 1D workgroups
    for (const x of [64, 128, 256, 512]) {
      if (x <= maxX && x <= maxInvocations) {
        candidates.push([x, 1, 1]);
      }
    }

    // 2D workgroups (for matrix operations)
    for (const x of [8, 16, 32]) {
      for (const y of [8, 16, 32]) {
        if (x <= maxX && y <= maxY && x * y <= maxInvocations) {
          candidates.push([x, y, 1]);
        }
      }
    }

    return candidates;
  }

  /**
   * Tune a kernel by running benchmarks
   * @param kernelName - Name of kernel to tune
   * @param inputSizes - Input dimensions for tuning
   * @param options - Tuning options
   * @returns Promise resolving to tuning result
   */
  async tuneKernel(
    kernelName: string,
    inputSizes: InputSizes,
    options: TuneConfig = {}
  ): Promise<TuneResult> {
    const {
      warmup = getTunerConfig().defaultWarmupIterations,
      iterations = getTunerConfig().defaultTimedIterations,
      forceRetune = false,
    } = options;

    // Check cache
    const cacheKey: CacheKey = generateCacheKey(kernelName, inputSizes);
    if (!forceRetune && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    // Get candidates to test
    const candidates = this._generateWorkgroupCandidates();

    // Run tuning based on kernel type
    let bestResult: TuneResult;

    if (!this.device) {
      return tuneGeneric(this.capabilities);
    }

    switch (kernelName) {
      case 'matmul':
        bestResult = await tuneMatmul(
          this.device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.capabilities
        );
        break;
      case 'attention':
        bestResult = await tuneAttention(
          this.device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.capabilities
        );
        break;
      case 'softmax':
        bestResult = await tuneSoftmax(
          this.device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.capabilities
        );
        break;
      case 'rmsnorm':
        bestResult = await tuneRMSNorm(
          this.device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.capabilities
        );
        break;
      case 'dequant':
        bestResult = await tuneDequant(
          this.device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.capabilities
        );
        break;
      default:
        bestResult = tuneGeneric(this.capabilities);
    }

    // Cache result
    this.cache.set(cacheKey, bestResult);
    saveCache(this.cache, this.capabilities);

    return bestResult;
  }

  /**
   * Get cached tuning result
   * @param kernelName - Kernel name
   * @param inputSizes - Input sizes
   * @returns Cached result or null
   */
  getCachedResult(kernelName: string, inputSizes: InputSizes): TuneResult | null {
    const cacheKey: CacheKey = generateCacheKey(kernelName, inputSizes);
    return this.cache.get(cacheKey) || null;
  }

  /**
   * Clear all cached results
   */
  clearCache(): void {
    this.cache.clear();
    clearCacheStorage(this.capabilities);
  }

  /**
   * Get all cached results
   * @returns Object with all cached results
   */
  getAllCachedResults(): Record<string, TuneRecord> {
    return Object.fromEntries(this.cache);
  }

  /**
   * Destroy tuner resources
   */
  destroy(): void {
    if (this.profiler) {
      this.profiler.destroy();
    }
  }
}

// Global tuner instance
let globalTuner: KernelTuner | null = null;

/**
 * Get the global kernel tuner
 * @returns Promise resolving to kernel tuner instance
 */
export async function getKernelTuner(): Promise<KernelTuner> {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}

/**
 * Convenience function to tune a kernel
 * @param kernelName - Kernel name
 * @param inputSizes - Input sizes
 * @returns Promise resolving to tuning result
 */
export async function tuneKernel(
  kernelName: string,
  inputSizes: InputSizes
): Promise<TuneResult> {
  const tuner = await getKernelTuner();
  return tuner.tuneKernel(kernelName, inputSizes);
}
