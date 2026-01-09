

// ============================================================================
// Re-exports from kernel-configs
// ============================================================================

export {
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
  createUniformBufferFromData,
  createUniformBufferWithView,
} from './uniform-utils.js';

// ============================================================================
// Combined Cache Management
// ============================================================================

import { clearShaderCaches, getShaderCacheStats } from './shader-cache.js';
import { clearPipelineCaches, getPipelineCacheStats } from './pipeline-cache.js';


export function clearKernelCaches() {
  clearShaderCaches();
  clearPipelineCaches();
}


export function clearPipelineCache() {
  clearKernelCaches();
}


export function getCacheStats() {
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
