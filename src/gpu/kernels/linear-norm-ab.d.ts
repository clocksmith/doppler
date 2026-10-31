import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface LinearNormABOptions {
  /**
   * f32 reproduces the unfused input norm and both projections bit for bit;
   * f32_single_pass reproduces the normalized row and scales single-pass row dots by the
   * inverse root.
   */
  variant: 'f32' | 'f32_single_pass';
  /** Layer input [1, hiddenSize] F32. */
  input: Tensor;
  /** Norm weight; its dtype resolves as rmsnorm.js resolves it. */
  normWeight: GPUBuffer | WeightBuffer;
  rmsNormWeightOffset: boolean;
  eps: number;
  /** in_proj_a [rows, hiddenSize] F32 row-major. */
  aWeight: GPUBuffer | WeightBuffer;
  /** in_proj_b [rows, hiddenSize] F32 row-major. */
  bWeight: GPUBuffer | WeightBuffer;
  hiddenSize: number;
  /** Value heads: the row count of each projection. */
  rows: number;
  /** Receives the normalized row (rmsnorm output). */
  normedOutput: GPUBuffer;
  aOutput: GPUBuffer;
  bOutput: GPUBuffer;
}

/**
 * Record the decode input RMSNorm of a linear-attention layer and its in_proj_a and
 * in_proj_b GEMVs of one token in one dispatch, with the results of rmsnorm/default
 * and matmul/gemv_f32_subgroup.
 */
export declare function recordLinearNormAB(recorder: CommandRecorder, options: LinearNormABOptions): Promise<void>;
