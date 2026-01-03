/**
 * Kernel Utilities - Shared utilities for kernel management
 *
 * This module re-exports utilities from specialized submodules:
 * - kernel-configs: Kernel configuration data
 * - shader-cache: Shader loading and compilation
 * - pipeline-cache: Pipeline creation and caching
 * - feature-check: Device capability checking
 * - kernel-tuning: Auto-tuning and prewarming
 * - uniform-utils: Uniform buffer helpers
 *
 * @module gpu/kernels/utils
 */

// ============================================================================
// Re-exports from kernel-configs
// ============================================================================

export {
  type VariantMetadata,
  type KernelConfig,
  KERNEL_CONFIGS,
  getKernelConfig,
  setKernelValidator,
} from './kernel-configs.js';

// ============================================================================
// Re-exports from shader-cache
// ============================================================================

export {
  loadShaderSource,
  compileShader,
  getShaderModule,
  clearShaderCaches,
  getShaderCacheStats,
} from './shader-cache.js';

// ============================================================================
// Re-exports from pipeline-cache
// ============================================================================

export {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
  getCachedPipeline,
  getPipelineFast,
  createPipeline,
  clearPipelineCaches,
  getPipelineCacheStats,
} from './pipeline-cache.js';

// ============================================================================
// Re-exports from feature-check
// ============================================================================

export {
  type FeatureCapabilities,
  hasRequiredFeatures,
  validateAttentionLimits,
} from './feature-check.js';

// ============================================================================
// Re-exports from kernel-tuning
// ============================================================================

export {
  getTunedWorkgroupSize,
  autoTuneKernels,
  prewarmKernels,
} from './kernel-tuning.js';

// ============================================================================
// Re-exports from uniform-utils
// ============================================================================

export {
  type UniformBufferOptions,
  createUniformBufferFromData,
  createUniformBufferWithView,
} from './uniform-utils.js';

// ============================================================================
// Combined Cache Management
// ============================================================================

/**
 * Clear all kernel caches
 */
export declare function clearKernelCaches(): void;

/**
 * Alias for clearKernelCaches for backward compatibility
 */
export declare function clearPipelineCache(): void;

/**
 * Get combined cache statistics
 */
export declare function getCacheStats(): {
  pipelines: number;
  shaders: number;
  shaderModules: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
};
