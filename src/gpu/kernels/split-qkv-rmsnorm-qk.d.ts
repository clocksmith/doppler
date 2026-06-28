import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { TensorLike, WeightBuffer } from '../weight-buffer.js';

export interface SplitQKVRMSNormQKOptions {
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  qSize: number;
  kSize: number;
  vSize: number;
  rmsNormWeightOffset?: boolean;
  skipKNorm?: boolean;
  allowUnitQKNorm?: boolean;
}

export interface SplitQKVRMSNormQKResult {
  Q: Tensor;
  K: Tensor;
  V: Tensor;
}

export function canUseSplitQKVRMSNormQK(
  qkvTensor: Tensor | null | undefined,
  options?: Pick<SplitQKVRMSNormQKOptions, 'skipKNorm' | 'allowUnitQKNorm'>
): boolean;

export function runSplitQKVRMSNormQK(
  qkvTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options: SplitQKVRMSNormQKOptions
): Promise<SplitQKVRMSNormQKResult>;

export function recordSplitQKVRMSNormQK(
  recorder: CommandRecorder,
  qkvTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options: SplitQKVRMSNormQKOptions
): Promise<SplitQKVRMSNormQKResult>;
