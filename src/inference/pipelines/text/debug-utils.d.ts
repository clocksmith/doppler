/**
 * Debug utilities for pipeline tracing.
 *
 * Toggleable log categories for surgical debugging without noise.
 * Enable via: setDebugCategories({ embed: true, layer: true })
 *
 * @module inference/pipelines/text/debug-utils
 */

// Public aggregation entrypoint
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
  DEBUG_PROFILES,
} from './debug-utils/index.js';
