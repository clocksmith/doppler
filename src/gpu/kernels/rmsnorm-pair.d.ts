import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { WeightBuffer, TensorLike } from '../weight-buffer.js';

export declare const RMSNORM_PAIR_CACHE_LIMIT: number;

export interface SandwichRMSNormPairOptions {
  batchSize?: number;
  hiddenSize?: number | null;
  rmsNormWeightOffset?: boolean;
  label?: string | null;
  layerIdx?: number;
  postOutputBuffer?: GPUBuffer | null;
  preOutputBuffer?: GPUBuffer | null;
}

export interface SandwichRMSNormPairResult {
  postAttn: Tensor;
  ffnInput: Tensor;
}

export declare function runSandwichRMSNormPair(
  input: Tensor,
  residual: Tensor | null,
  postWeight: GPUBuffer | WeightBuffer | TensorLike,
  preWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options?: SandwichRMSNormPairOptions
): Promise<SandwichRMSNormPairResult>;

export declare function recordSandwichRMSNormPair(
  recorder: CommandRecorder,
  input: Tensor,
  residual: Tensor | null,
  postWeight: GPUBuffer | WeightBuffer | TensorLike,
  preWeight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options?: SandwichRMSNormPairOptions
): Promise<SandwichRMSNormPairResult>;
