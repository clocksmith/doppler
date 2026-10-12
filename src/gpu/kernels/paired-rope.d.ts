import type { RoPEPrecomputeOptions } from './rope-precompute.js';
/** Paired table generation: declared ordinary/linear scaling and source frequency geometry. */
export declare function runPairedRoPEPrecompute(options: RoPEPrecomputeOptions): Promise<{ cos: GPUBuffer; sin: GPUBuffer }>;
