

// Public aggregation entrypoint.
// Implementation is split into debug-utils/ submodules.

export {
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
