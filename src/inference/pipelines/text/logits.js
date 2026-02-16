

export {
  // CPU functions
  rmsNormCPU,
  matmulCPU,
  applySoftcapping,
  // GPU functions
  computeLogitsGPU,
  recordLogitsGPU,
  // Utilities
  extractLastPositionLogits,
  finalizeLogits,
  // Main orchestrator
  computeLogits,
} from './logits/index.js';
