/**
 * Debug utilities for pipeline tracing.
 *
 * Re-exports all debug utilities from submodules.
 *
 * @module inference/pipelines/text/debug-utils
 */

// Configuration
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
  type TensorStats,
  dumpTensor,
  dumpTokenVector,
  dumpKVCache,
  logKernelStep,
  isKernelDebugEnabled,
} from './tensor.js';

export { getBufferStats, getLogitsHealth } from './health-metrics.js';
export { DEBUG_PROFILES } from './profiles.js';
export { decodeReadback, f16ToF32 } from './readback-decoding.js';
