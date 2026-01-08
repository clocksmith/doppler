/**
 * Kernel Auto-Tuner - Optimal Workgroup Size Selection
 *
 * Re-export facade for backward compatibility.
 * Implementation has been split into focused submodules in ./kernel-tuner/
 *
 * @see ./kernel-tuner/types.ts - Type definitions
 * @see ./kernel-tuner/cache.ts - LocalStorage caching
 * @see ./kernel-tuner/benchmarks.ts - Kernel benchmark functions
 * @see ./kernel-tuner/tuner.ts - Main KernelTuner class
 */

export {
  // Main exports
  KernelTuner,
  getKernelTuner,
  tuneKernel,

  // Default export
  default,
} from './kernel-tuner/index.js';
