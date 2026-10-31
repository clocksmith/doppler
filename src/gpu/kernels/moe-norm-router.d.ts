import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface MoENormRouterOptions {
  /** Attention output [1, hiddenSize] F32. */
  input: Tensor;
  /** Layer input the attention output is added to before the norm [1, hiddenSize] F32. */
  residual: Tensor;
  /** Norm weight; its dtype resolves as rmsnorm.js resolves it. */
  normWeight: GPUBuffer | WeightBuffer;
  rmsNormWeightOffset: boolean;
  eps: number;
  /** Expert router [numExperts, hiddenSize] F32 row-major. */
  routerWeight: GPUBuffer | WeightBuffer;
  /** Shared-expert gate [1, hiddenSize] F32. */
  sharedRouterWeight: GPUBuffer | WeightBuffer;
  hiddenSize: number;
  numExperts: number;
  /** Receives the normalized row (rmsnorm output). */
  normedOutput: GPUBuffer;
  /** Receives input + residual (rmsnorm OUTPUT_PRENORM). */
  prenormOutput: GPUBuffer;
  /** Receives numExperts router logits. */
  logitsOutput: GPUBuffer;
  /** Receives the shared-expert gate logit. */
  sharedGateOutput: GPUBuffer;
}

/**
 * Record the decode post-attention RMSNorm (folded attention residual) and the
 * expert-router and shared-expert router GEMVs of one token in one dispatch, with the
 * results of rmsnorm/default and matmul/gemv_f32_subgroup.
 */
export declare function recordMoENormRouter(recorder: CommandRecorder, options: MoENormRouterOptions): Promise<void>;
