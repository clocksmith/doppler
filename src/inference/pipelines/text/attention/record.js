/**
 * Recorded attention facade.
 *
 * Semantic decisions are resolved in plan.js. The recorded adapter binds the
 * canonical interpreter to the caller's recorder.
 */

export { recordLayerAttentionGPU } from './executor-recorded.js';
