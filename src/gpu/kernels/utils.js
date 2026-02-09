

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
  writeUniformsFromObject,
} from './uniform-utils.js';

// ============================================================================
// Unified Kernel Helper
// ============================================================================

import { getKernelConfig } from './kernel-configs.js';
import { getPipelineFast } from './pipeline-cache.js';
import { getDevice } from '../device.js';
import { dispatchKernel, dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { createUniformBufferWithView as createUniformBuffer } from './uniform-utils.js';
import { writeUniformsFromObject } from './uniform-utils.js';

export async function unifiedKernelWrapper(opName, target, variant, bindings, uniforms, workgroups, constants = null) {
  const device = target?.device || getDevice();
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const config = getKernelConfig(opName, variant);
  const pipeline = await getPipelineFast(opName, variant, null, constants);

  const uniformBuffer = createUniformBuffer(
    `${opName}_uniforms`,
    config.uniforms.size,
    (view) => writeUniformsFromObject(view, config, uniforms),
    recorder,
    device
  );

  const bindGroupEntries = [
    { binding: 0, resource: { buffer: uniformBuffer } }
  ];

  const dataBindings = config.bindings
    .filter(b => b.type !== 'uniform')
    .slice()
    .sort((a, b) => a.index - b.index);

  if (bindings.length !== dataBindings.length) {
    throw new Error(
      `Kernel "${opName}/${variant}" expected ${dataBindings.length} bindings ` +
      `(excluding uniforms) but got ${bindings.length}`
    );
  }

  for (let i = 0; i < bindings.length; i++) {
    const binding = bindings[i];
    const bindingConfig = dataBindings[i];
    let index = bindingConfig.index;

    // Some variants change output binding index (e.g. gather f16 output uses binding 4).
    if (bindingConfig.name === 'output' && config.variantMetadata?.outputBinding != null) {
      index = config.variantMetadata.outputBinding;
    }

    bindGroupEntries.push({
      binding: index,
      resource: { buffer: binding?.buffer || binding }
    });
  }

  const bindGroup = device.createBindGroup({
    label: `${opName}_bind_group`,
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  if (workgroups && typeof workgroups === 'object' && workgroups.indirectBuffer) {
    const indirectOffset = workgroups.indirectOffset ?? 0;
    if (recorder) {
      recordDispatchIndirect(recorder, pipeline, bindGroup, workgroups.indirectBuffer, indirectOffset, opName);
    } else {
      dispatchIndirect(device, pipeline, bindGroup, workgroups.indirectBuffer, indirectOffset, opName);
    }
  } else {
    dispatchKernel(target, pipeline, bindGroup, workgroups, opName);
  }

  if (!recorder) {
    uniformBuffer.destroy();
  }

  return true;
}

// ============================================================================
// Debug Helpers
// ============================================================================

import { log, isTraceEnabled } from '../../debug/index.js';


export async function createBindGroupWithValidation(device, descriptor, contextLabel) {
  if (!isTraceEnabled('buffers')) {
    return device.createBindGroup(descriptor);
  }

  device.pushErrorScope('validation');
  const bindGroup = device.createBindGroup(descriptor);
  const error = await device.popErrorScope();
  if (error) {
    log.error('Kernels', `${contextLabel} bindGroup validation: ${error.message}`);
  }
  return bindGroup;
}

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
setKernelValidator('attention', 'prefill_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming_f16kv', validateAttentionLimits);
