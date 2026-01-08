/**
 * Logits computation - final layer norm and LM head projection.
 *
 * This is a re-export facade for backward compatibility.
 * Implementation has been split into focused modules under logits/.
 *
 * @module inference/pipeline/logits
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
  // Main orchestrator
  computeLogits,
} from './logits/index.js';
