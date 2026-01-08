/**
 * Attention Module - Re-export facade
 *
 * This file re-exports from the attention/ directory for backward compatibility.
 * New code should import directly from 'inference/pipelines/text/attention/index.js'.
 *
 * @module inference/pipelines/text/attention
 */

export {
  // Types
  type AttentionConfig,
  type AttentionState,
  type AttentionResult,
  type AttentionDebugFlags,
  // Utilities
  shouldDebugLayer,
  markStageLogged,
  releaseOrTrack,
  getQKNormOnesBuffer,
  // Functions
  runLayerAttentionGPU,
  recordLayerAttentionGPU,
} from './attention/index.js';
