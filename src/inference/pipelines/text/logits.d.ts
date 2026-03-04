/**
 * Logits computation - final layer norm and LM head projection.
 *
 * Public aggregation entrypoint.
 * Implementation is split into focused modules under logits/.
 *
 * @module inference/pipelines/text/logits
 */

export {
  // Types
  type LogitsConfig,
  type LogitsWeights,
  type LogitsDebugFlags,
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
