

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
  // Main orchestrator
  computeLogits,
} from './logits/index.js';
