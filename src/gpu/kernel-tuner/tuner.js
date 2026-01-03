

import { getDevice, getKernelCapabilities, getDeviceLimits } from '../device.js';
import { getKernelThresholds } from '../../config/schema/index.js';
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


export class KernelTuner {
  
  #device;

  
  #profiler;

  
  #limits;

  
  #capabilities;

  
  #cache;

  constructor() {
    this.#device = null;
    this.#profiler = null;
    this.#limits = null;
    this.#capabilities = null;
    this.#cache = new Map();
  }

  
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

  
  #generateWorkgroupCandidates() {
    const { maxComputeWorkgroupSizeX, maxComputeWorkgroupSizeY, maxComputeInvocationsPerWorkgroup } = getKernelThresholds().tuner;
    const maxX = this.#limits?.maxComputeWorkgroupSizeX ?? maxComputeWorkgroupSizeX;
    const maxY = this.#limits?.maxComputeWorkgroupSizeY ?? maxComputeWorkgroupSizeY;
    const maxInvocations = this.#limits?.maxComputeInvocationsPerWorkgroup ?? maxComputeInvocationsPerWorkgroup;

    
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
    
    const cacheKey = generateCacheKey(kernelName, inputSizes);
    if (!forceRetune && this.#cache.has(cacheKey)) {
      return  (this.#cache.get(cacheKey));
    }

    // Get candidates to test
    const candidates = this.#generateWorkgroupCandidates();

    // Run tuning based on kernel type
    
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

  
  getCachedResult(kernelName, inputSizes) {
    
    const cacheKey = generateCacheKey(kernelName, inputSizes);
    return this.#cache.get(cacheKey) || null;
  }

  
  clearCache() {
    this.#cache.clear();
    clearCacheStorage(this.#capabilities);
  }

  
  getAllCachedResults() {
    return Object.fromEntries(this.#cache);
  }

  
  destroy() {
    if (this.#profiler) {
      this.#profiler.destroy();
    }
  }
}

// Global tuner instance

let globalTuner = null;


export async function getKernelTuner() {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}


export async function tuneKernel(
  kernelName,
  inputSizes
) {
  const tuner = await getKernelTuner();
  return tuner.tuneKernel(kernelName, inputSizes);
}
