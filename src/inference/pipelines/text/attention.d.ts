/**
 * Attention Module - Public aggregation entrypoint
 *
 * This file re-exports the attention public API from the attention/ directory.
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
