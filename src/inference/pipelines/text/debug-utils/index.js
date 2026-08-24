

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

export { getBufferStats, getLogitsHealth } from './health-metrics.js';
export { DEBUG_PROFILES } from './profiles.js';
export { decodeReadback, f16ToF32 } from './readback-decoding.js';
