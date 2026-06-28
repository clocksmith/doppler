import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { TensorLike, WeightBuffer } from '../weight-buffer.js';

export interface SplitQKVRMSNormRoPEQKOptions {
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  qSize: number;
  kSize: number;
  vSize: number;
  startPos?: number;
  rotaryDim?: number;
  pairSpanDim?: number;
  interleaved?: boolean;
  rmsNormWeightOffset?: boolean;
  reusesSharedKV?: boolean;
  skipKNorm?: boolean;
  allowUnitQKNorm?: boolean;
  f16KVCacheWrite?: {
    keysBuffer: GPUBuffer;
    valuesBuffer: GPUBuffer;
    dstOffset: number;
  } | null;
}

export interface SplitQKVRMSNormRoPEQKResult {
  Q: Tensor;
  K: Tensor | null;
  V: Tensor | null;
  wroteF16KVCache?: boolean;
}

export function canUseSplitQKVRMSNormRoPEQK(
  qkvTensor: Tensor | null | undefined,
  options?: Partial<SplitQKVRMSNormRoPEQKOptions>
): boolean;

export function runSplitQKVRMSNormRoPEQK(
  qkvTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  freqsCos: GPUBuffer | Tensor,
  freqsSin: GPUBuffer | Tensor,
  eps: number,
  options: SplitQKVRMSNormRoPEQKOptions
): Promise<SplitQKVRMSNormRoPEQKResult>;

export function recordSplitQKVRMSNormRoPEQK(
  recorder: CommandRecorder,
  qkvTensor: Tensor,
  qWeight: GPUBuffer | WeightBuffer | TensorLike,
  kWeight: GPUBuffer | WeightBuffer | TensorLike,
  freqsCos: GPUBuffer | Tensor,
  freqsSin: GPUBuffer | Tensor,
  eps: number,
  options: SplitQKVRMSNormRoPEQKOptions
): Promise<SplitQKVRMSNormRoPEQKResult>;
