/**
 * Immediate attention facade.
 *
 * Semantic decisions are resolved in plan.js. The immediate adapter records
 * the canonical interpreter and submits it immediately.
 */

export { runLayerAttentionGPU } from './executor-immediate.js';
