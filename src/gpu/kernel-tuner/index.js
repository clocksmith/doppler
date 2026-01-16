

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
