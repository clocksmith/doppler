

const __DBG_REC = { byOp: new Map(), totalCount: 0, total: { pipeline: 0, prep: 0, bg: 0, dispatch: 0 }, firstCallByOp: new Map() };
export function __dbgRecord(op, variant, pipelineMs, prepMs, bgMs, dispatchMs) {
  const key = `${op}/${variant}`;
  const e = __DBG_REC.byOp.get(key) ?? { count: 0, pipeline: 0, prep: 0, bg: 0, dispatch: 0, firstPipeline: 0 };
  if (e.count === 0) e.firstPipeline = pipelineMs;
  e.count += 1;
  e.pipeline += pipelineMs;
  e.prep += prepMs;
  e.bg += bgMs;
  e.dispatch += dispatchMs;
  __DBG_REC.byOp.set(key, e);
  __DBG_REC.totalCount += 1;
  __DBG_REC.total.pipeline += pipelineMs;
  __DBG_REC.total.prep += prepMs;
  __DBG_REC.total.bg += bgMs;
  __DBG_REC.total.dispatch += dispatchMs;
}
if (typeof process !== "undefined" && process?.env?.DOPPLER_DBG_RECORD === '1') {
  process.on('exit', () => {
    const t = __DBG_REC.total;
    const sum = t.pipeline + t.prep + t.bg + t.dispatch;
    process.stderr.write(`\n[DBG_REC] total records=${__DBG_REC.totalCount} sum=${sum.toFixed(1)}ms (pipeline=${t.pipeline.toFixed(1)} prep=${t.prep.toFixed(1)} bg=${t.bg.toFixed(1)} dispatch=${t.dispatch.toFixed(1)})\n`);
    const rows = Array.from(__DBG_REC.byOp.entries()).map(([k, v]) => ({ k, ...v, totalMs: v.pipeline + v.prep + v.bg + v.dispatch }));
    rows.sort((a, b) => b.totalMs - a.totalMs);
    process.stderr.write(`[DBG_REC] per kernel (top 25):\n`);
    for (const r of rows.slice(0, 25)) {
      process.stderr.write(`  ${r.k.padEnd(45)} n=${String(r.count).padStart(5)} total=${r.totalMs.toFixed(1).padStart(7)}ms pipeline=${r.pipeline.toFixed(1).padStart(7)} prep=${r.prep.toFixed(1).padStart(6)} bg=${r.bg.toFixed(1).padStart(6)} dispatch=${r.dispatch.toFixed(1).padStart(7)} firstPipe=${r.firstPipeline.toFixed(2)}\n`);
    }
  });
}


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
  getUniformByteLength,
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
import { getUniformByteLength } from './uniform-utils.js';
import { writeUniformsFromObject } from './uniform-utils.js';

export async function unifiedKernelWrapper(opName, target, variant, bindings, uniforms, workgroups, constants = null, extraBindings = null) {
  const __dbg = (typeof process !== "undefined" && process?.env?.DOPPLER_DBG_RECORD === '1');
  const __t0 = __dbg ? performance.now() : 0;
  const device = target?.device || getDevice();
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const config = getKernelConfig(opName, variant);
  const pipeline = await getPipelineFast(opName, variant, null, constants);
  const __tPipeline = __dbg ? performance.now() : 0;

  const uniformBuffer = createUniformBuffer(
    `${opName}_uniforms`,
    getUniformByteLength(config),
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

    const buffer = binding?.buffer || binding;
    const isGpuBuffer = buffer && (
      typeof GPUBuffer === 'undefined'
        ? true
        : buffer instanceof GPUBuffer
    );
    if (!isGpuBuffer) {
      const bindingLabel = binding?.label ?? buffer?.label ?? 'unknown';
      const bufferType = buffer === null ? 'null' : buffer === undefined ? 'undefined' : buffer.constructor?.name || typeof buffer;
      throw new Error(
        `Kernel "${opName}/${variant}" binding "${bindingConfig.name}" (index ${index}) requires a GPUBuffer ` +
        `(label=${bindingLabel}, type=${bufferType}).`
      );
    }

    bindGroupEntries.push({
      binding: index,
      resource: { buffer }
    });
  }

  // Append extra bindings not tracked in registry (e.g. OUTPUT_PRENORM residual_sum_output)
  if (extraBindings) {
    for (const extra of extraBindings) {
      const buf = extra.buffer?.buffer || extra.buffer;
      bindGroupEntries.push({
        binding: extra.binding,
        resource: { buffer: buf },
      });
    }
  }

  try {
    const __tBgStart = __dbg ? performance.now() : 0;
    const bindGroup = device.createBindGroup({
      label: `${opName}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries,
    });
    const __tBg = __dbg ? performance.now() : 0;

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
    if (__dbg) {
      const __tEnd = performance.now();
      __dbgRecord(opName, variant, __tPipeline - __t0, __tBgStart - __tPipeline, __tBg - __tBgStart, __tEnd - __tBg);
    }
  } catch (error) {
    if (!recorder) {
      uniformBuffer.destroy();
    }
    throw error;
  }

  if (!recorder) {
    device.queue.onSubmittedWorkDone()
      .then(() => {
        uniformBuffer.destroy();
      })
      .catch(() => {
        uniformBuffer.destroy();
      });
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
setKernelValidator('attention', 'decode_small', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming', validateAttentionLimits);
setKernelValidator('attention', 'prefill_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small_f16', validateAttentionLimits);
setKernelValidator('attention', 'decode_small_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming_f16', validateAttentionLimits);
setKernelValidator('attention', 'prefill_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_small_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'decode_small_f16kv', validateAttentionLimits);
setKernelValidator('attention', 'prefill_streaming_f16kv', validateAttentionLimits);
