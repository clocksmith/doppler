/**
 * Kernel Tuner - Main Class
 *
 * Orchestrates kernel auto-tuning by running benchmarks with various
 * workgroup configurations and caching optimal results.
 */

import { getDevice, getKernelCapabilities, getDeviceLimits } from '../device.js';
import { GPUProfiler } from '../profiler.js';
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
  /** @type {GPUDevice | null} */
  #device;

  /** @type {import('../profiler.js').GPUProfiler | null} */
  #profiler;

  /** @type {import('./types.js').DeviceLimits | null} */
  #limits;

  /** @type {import('./types.js').KernelCapabilities | null} */
  #capabilities;

  /** @type {Map<import('./types.js').CacheKey, import('./types.js').TuneRecord>} */
  #cache;

  constructor() {
    this.#device = null;
    this.#profiler = null;
    this.#limits = null;
    this.#capabilities = null;
    this.#cache = new Map();
  }

  /**
   * Initialize the tuner
   * @returns {Promise<void>}
   */
  async init() {
    this.#device = getDevice();
    if (!this.#device) {
      throw new Error('GPU device not initialized');
    }

    this.#profiler = new GPUProfiler(this.#device);
    this.#limits = getDeviceLimits();
    this.#capabilities = getKernelCapabilities();

    // Load cached results
    this.#cache = loadCache(this.#capabilities);
  }

  /**
   * Generate workgroup size candidates based on device limits
   * @returns {import('./types.js').WorkgroupSize[]}
   */
  #generateWorkgroupCandidates() {
    const maxX = this.#limits?.maxComputeWorkgroupSizeX || 256;
    const maxY = this.#limits?.maxComputeWorkgroupSizeY || 256;
    const maxInvocations = this.#limits?.maxComputeInvocationsPerWorkgroup || 256;

    /** @type {import('./types.js').WorkgroupSize[]} */
    const candidates = [];

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
   * @param {string} kernelName - Name of kernel to tune
   * @param {import('./types.js').InputSizes} inputSizes - Input dimensions for tuning
   * @param {import('./types.js').TuneConfig} [options] - Tuning options
   * @returns {Promise<import('./types.js').TuneResult>} Promise resolving to tuning result
   */
  async tuneKernel(
    kernelName,
    inputSizes,
    options = {}
  ) {
    const {
      warmup = getTunerConfig().defaultWarmupIterations,
      iterations = getTunerConfig().defaultTimedIterations,
      forceRetune = false,
    } = options;

    // Check cache
    /** @type {import('./types.js').CacheKey} */
    const cacheKey = generateCacheKey(kernelName, inputSizes);
    if (!forceRetune && this.#cache.has(cacheKey)) {
      return /** @type {import('./types.js').TuneResult} */ (this.#cache.get(cacheKey));
    }

    // Get candidates to test
    const candidates = this.#generateWorkgroupCandidates();

    // Run tuning based on kernel type
    /** @type {import('./types.js').TuneResult} */
    let bestResult;

    if (!this.#device) {
      return tuneGeneric(this.#capabilities);
    }

    switch (kernelName) {
      case 'matmul':
        bestResult = await tuneMatmul(
          this.#device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.#capabilities
        );
        break;
      case 'attention':
        bestResult = await tuneAttention(
          this.#device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.#capabilities
        );
        break;
      case 'softmax':
        bestResult = await tuneSoftmax(
          this.#device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.#capabilities
        );
        break;
      case 'rmsnorm':
        bestResult = await tuneRMSNorm(
          this.#device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.#capabilities
        );
        break;
      case 'dequant':
        bestResult = await tuneDequant(
          this.#device,
          inputSizes,
          candidates,
          warmup,
          iterations,
          this.#capabilities
        );
        break;
      default:
        bestResult = tuneGeneric(this.#capabilities);
    }

    // Cache result
    this.#cache.set(cacheKey, bestResult);
    saveCache(this.#cache, this.#capabilities);

    return bestResult;
  }

  /**
   * Get cached tuning result
   * @param {string} kernelName - Kernel name
   * @param {import('./types.js').InputSizes} inputSizes - Input sizes
   * @returns {import('./types.js').TuneResult | null} Cached result or null
   */
  getCachedResult(kernelName, inputSizes) {
    /** @type {import('./types.js').CacheKey} */
    const cacheKey = generateCacheKey(kernelName, inputSizes);
    return this.#cache.get(cacheKey) || null;
  }

  /**
   * Clear all cached results
   * @returns {void}
   */
  clearCache() {
    this.#cache.clear();
    clearCacheStorage(this.#capabilities);
  }

  /**
   * Get all cached results
   * @returns {Record<string, import('./types.js').TuneRecord>} Object with all cached results
   */
  getAllCachedResults() {
    return Object.fromEntries(this.#cache);
  }

  /**
   * Destroy tuner resources
   * @returns {void}
   */
  destroy() {
    if (this.#profiler) {
      this.#profiler.destroy();
    }
  }
}

// Global tuner instance
/** @type {KernelTuner | null} */
let globalTuner = null;

/**
 * Get the global kernel tuner
 * @returns {Promise<KernelTuner>} Promise resolving to kernel tuner instance
 */
export async function getKernelTuner() {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}

/**
 * Convenience function to tune a kernel
 * @param {string} kernelName - Kernel name
 * @param {import('./types.js').InputSizes} inputSizes - Input sizes
 * @returns {Promise<import('./types.js').TuneResult>} Promise resolving to tuning result
 */
export async function tuneKernel(
  kernelName,
  inputSizes
) {
  const tuner = await getKernelTuner();
  return tuner.tuneKernel(kernelName, inputSizes);
}
