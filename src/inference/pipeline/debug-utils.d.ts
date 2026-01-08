/**
 * Debug utilities for pipeline tracing.
 *
 * Toggleable log categories for surgical debugging without noise.
 * Enable via: setDebugCategories({ embed: true, layer: true })
 *
 * @module inference/pipeline/debug-utils
 */

// Re-export facade for backward compatibility
export {
  type DebugCategory,
  type DebugConfig,
  setDebugCategories,
  resetDebugConfig,
  applyPipelineDebugConfig,
  getDebugConfig,
  incrementDecodeStep,
  resetDecodeStep,
  getDecodeStep,
  shouldDebugLayerOutput,
  logEmbed,
  logLayer,
  logAttn,
  logFFN,
  logKV,
  logLogits,
  logSample,
  logIO,
  logPerf,
  type TensorStats,
  dumpTensor,
  dumpTokenVector,
  dumpKVCache,
  logKernelStep,
  isKernelDebugEnabled,
  f16ToF32,
  decodeReadback,
  getLogitsHealth,
  getBufferStats,
  DEBUG_PRESETS,
} from './debug-utils/index.js';
