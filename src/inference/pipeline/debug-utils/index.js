/**
 * Debug utilities for pipeline tracing.
 *
 * Re-exports all debug utilities from submodules.
 *
 * @module inference/pipeline/debug-utils
 */

// Configuration
export {
  setDebugCategories,
  resetDebugConfig,
  applyPipelineDebugConfig,
  getDebugConfig,
  incrementDecodeStep,
  resetDecodeStep,
  getDecodeStep,
  shouldDebugLayerOutput,
} from './config.js';

// Logging
export {
  logEmbed,
  logLayer,
  logAttn,
  logFFN,
  logKV,
  logLogits,
  logSample,
  logIO,
  logPerf,
} from './logging.js';

// Tensor inspection
export {
  dumpTensor,
  dumpTokenVector,
  dumpKVCache,
  logKernelStep,
  isKernelDebugEnabled,
} from './tensor.js';

// Utilities
export {
  f16ToF32,
  decodeReadback,
  getLogitsHealth,
  getBufferStats,
  DEBUG_PRESETS,
} from './utils.js';
