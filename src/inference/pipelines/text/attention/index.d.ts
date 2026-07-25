/**
 * Attention Module - Re-exports
 *
 * @module inference/pipelines/text/attention
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
export {
  ATTENTION_PLAN_SCHEMA,
  type SemanticAttentionPlan,
  resolveAttentionPlan,
  bindAttentionPlan,
  executeBoundAttentionPlan,
  createAttentionExecutor,
} from './plan.js';
export { runAttentionBDPA, recordAttentionBDPA } from '../../../../gpu/kernel-selector.js';
