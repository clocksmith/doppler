import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { WeightBuffer, TensorLike } from '../weight-buffer.js';

export interface RMSNormQKOptions {
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  rmsNormWeightOffset?: boolean;
  /**
   * Paired (two-F32) input only: the execution graph whose `q_norm` and `k_norm` steps declare
   * the head-normalization programs, and the phase and layer they run in. Required for paired
   * input; scalar input selects the fused scalar program by rule.
   */
  kernelPath?: import('../../config/schema/kernel-path.schema.js').KernelPathSchema | null;
  phase?: 'prefill' | 'decode';
  layerIdx?: number;
}

export interface RMSNormQKResult {
  q: Tensor;
  k: Tensor;
}

export function canUseRMSNormQK(
  qTensor: Tensor | null | undefined,
  kTensor: Tensor | null | undefined,
  options?: { skipKNorm?: boolean }
): boolean;

export declare function runRMSNormQK(
  qTensor: Tensor,
  kTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options: RMSNormQKOptions
): Promise<RMSNormQKResult>;

export declare function recordRMSNormQK(
  recorder: CommandRecorder,
  qTensor: Tensor,
  kTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options: RMSNormQKOptions
): Promise<RMSNormQKResult>;
