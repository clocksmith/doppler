/**
 * Attention Module - Re-exports
 *
 * @module inference/pipeline/attention
 */

// Types and utilities
export {
  type AttentionConfig,
  type AttentionState,
  type AttentionResult,
  type AttentionDebugFlags,
  shouldDebugLayer,
  markStageLogged,
  releaseOrTrack,
  getQKNormOnesBuffer,
} from './types.js';

// Run (immediate submission)
export { runLayerAttentionGPU } from './run.js';

// Record (batched submission)
export { recordLayerAttentionGPU } from './record.js';
