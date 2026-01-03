/**
 * Split QKV Kernel
 *
 * Splits fused QKV projection output into separate Q, K, V buffers.
 * Used for 3→1 matmul optimization in attention.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';

/** Split QKV options */
export interface SplitQKVOptions {
  numTokens: number;
  qSize: number;  // numHeads * headDim
  kSize: number;  // numKVHeads * headDim
  vSize: number;  // numKVHeads * headDim
  /** Pre-allocated Q output tensor */
  qTensor?: Tensor | null;
  /** Pre-allocated K output tensor */
  kTensor?: Tensor | null;
  /** Pre-allocated V output tensor */
  vTensor?: Tensor | null;
}

/** Split QKV result */
export interface SplitQKVResult {
  Q: Tensor;
  K: Tensor;
  V: Tensor;
}

/**
 * Split fused QKV output into separate Q, K, V tensors.
 *
 * @param qkvTensor - Fused QKV output [numTokens, qSize + kSize + vSize]
 * @param options - Split configuration
 * @returns Separate Q, K, V tensors
 */
export declare function runSplitQKV(
  qkvTensor: Tensor,
  options: SplitQKVOptions
): Promise<SplitQKVResult>;

/**
 * Record split QKV (batched, no submit).
 */
export declare function recordSplitQKV(
  recorder: CommandRecorder,
  qkvTensor: Tensor,
  options: SplitQKVOptions
): Promise<SplitQKVResult>;
