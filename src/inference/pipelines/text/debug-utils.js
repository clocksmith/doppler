

// Re-export facade for backward compatibility
// Implementation split into debug-utils/ submodules

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
  DEBUG_PRESETS,
} from './debug-utils/index.js';
