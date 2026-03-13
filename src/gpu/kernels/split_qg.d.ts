/**
 * Split Q and Gate Kernel
 *
 * De-interleaves Q and Gate projections from q_proj output for attentionOutputGate models.
 * Models like Qwen 3.5 store q_proj weights in per-head interleaved layout:
 *   rows [h*headDim*2 : h*headDim*2+headDim] = Q for head h
 *   rows [h*headDim*2+headDim : (h+1)*headDim*2] = Gate for head h
 * This kernel separates the full matmul output into contiguous Q and Gate tensors.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';

/** Split Q and Gate options */
export interface SplitQGOptions {
  numTokens: number;
  numHeads: number;
  headDim: number;
  /** Pre-allocated Q output tensor */
  qTensor?: Tensor | null;
  /** Pre-allocated Gate output tensor */
  gTensor?: Tensor | null;
}

/** Split Q and Gate result */
export interface SplitQGResult {
  Q: Tensor;
  G: Tensor;
}

/**
 * De-interleave Q and Gate from q_proj output.
 *
 * @param qgTensor - Full q_proj output [numTokens, numHeads * headDim * 2] (interleaved)
 * @param options - Split configuration
 * @returns Separate Q and Gate tensors, each [numTokens, numHeads * headDim]
 */
export declare function runSplitQG(
  qgTensor: Tensor,
  options: SplitQGOptions
): Promise<SplitQGResult>;

/**
 * Record split Q and Gate (batched, no submit).
 */
export declare function recordSplitQG(
  recorder: CommandRecorder,
  qgTensor: Tensor,
  options: SplitQGOptions
): Promise<SplitQGResult>;
