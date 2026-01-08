/**
 * Kernel Auto-Tuner Module
 *
 * Re-exports all public APIs for the kernel tuner system.
 * This module automatically finds optimal workgroup sizes for different kernels
 * by running benchmarks with various configurations.
 *
 * Results are cached in localStorage for persistence across sessions.
 */

// Cache utilities
export {
  getTunerConfig,
  getDeviceSignature,
  generateCacheKey,
  loadCache,
  saveCache,
  clearCacheStorage,
} from './cache.js';

// Benchmark functions
export {
  benchmarkPipeline,
  createComputePipeline,
  tuneMatmul,
  tuneAttention,
  tuneSoftmax,
  tuneRMSNorm,
  tuneDequant,
  tuneGeneric,
  createMatmulShader,
  createAttentionShader,
  createSoftmaxShader,
  createRMSNormShader,
  createDequantShader,
} from './benchmarks.js';

// Main tuner class and utilities
export {
  KernelTuner,
  getKernelTuner,
  tuneKernel,
} from './tuner.js';

// Default export
export { KernelTuner as default } from './tuner.js';
