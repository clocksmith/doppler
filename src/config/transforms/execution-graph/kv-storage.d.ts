import type { ExecutionGraphTransform } from '../execution-graph-transforms.js';
/** Widen only attention KV readers, preserving stored weights and projections. */
export declare const widenAttentionKvToF32: ExecutionGraphTransform;
