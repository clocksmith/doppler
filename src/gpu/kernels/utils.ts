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

import { clearShaderCaches, getShaderCacheStats } from './shader-cache.js';
import { clearPipelineCaches, getPipelineCacheStats } from './pipeline-cache.js';

/**
 * Clear all kernel caches
 */
export function clearKernelCaches(): void {
  clearShaderCaches();
  clearPipelineCaches();
}

/**
 * Alias for clearKernelCaches for backward compatibility
 */
export function clearPipelineCache(): void {
  clearKernelCaches();
}

/**
 * Get combined cache statistics
 */
export function getCacheStats(): {
  pipelines: number;
  shaders: number;
  shaderModules: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
} {
  const shaderStats = getShaderCacheStats();
  const pipelineStats = getPipelineCacheStats();
  return {
    pipelines: pipelineStats.pipelines,
    shaders: shaderStats.sources,
    shaderModules: shaderStats.modules,
    bindGroupLayouts: pipelineStats.bindGroupLayouts,
    pipelineLayouts: pipelineStats.pipelineLayouts,
  };
}

// ============================================================================
// Attention Validator Initialization
// ============================================================================

import { setKernelValidator } from './kernel-configs.js';
import { validateAttentionLimits } from './feature-check.js';

// Set validators on attention configs that need them
// This avoids circular dependencies between configs and validation
setKernelValidator('attention', 'prefill', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming', validateAttentionLimits);
setKernelValidator('attention', 'prefill_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming_f16kv', validateAttentionLimits);
