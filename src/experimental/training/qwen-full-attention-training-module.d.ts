import type { Tensor } from '../../gpu/tensor.js';

export interface QwenFullAttentionTrainingModuleOptions {
  seqLen: number;
  hiddenSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  rotaryDim: number;
  pairSpanDim: number;
  interleaved?: boolean;
  startPos?: number;
  rmsEps: number;
}

export interface QwenFullAttentionTrainingModuleInputs {
  hidden: Tensor;
  qWeight: Tensor;
  kWeight: Tensor;
  vWeight: Tensor;
  oWeight: Tensor;
  qNormWeight: Tensor;
  kNormWeight: Tensor;
  cos: Tensor;
  sin: Tensor;
}

export interface QwenFullAttentionTrainingModuleCache {
  dims: Record<string, number | boolean>;
  rawQuery: Tensor;
  gate: Tensor;
  rawKey: Tensor;
  value: Tensor;
  queryRope: Tensor;
  keyRope: Tensor;
  attention: Tensor;
}

export declare function runQwenFullAttentionTrainingModuleForward(
  inputs: QwenFullAttentionTrainingModuleInputs,
  options: QwenFullAttentionTrainingModuleOptions
): Promise<{ output: Tensor; cache: QwenFullAttentionTrainingModuleCache }>;

export declare function runQwenFullAttentionTrainingModuleBackward(
  inputs: QwenFullAttentionTrainingModuleInputs,
  gradOutput: Tensor,
  cache: QwenFullAttentionTrainingModuleCache,
  options: QwenFullAttentionTrainingModuleOptions
): Promise<{ hidden: Tensor }>;

export declare function releaseQwenFullAttentionTrainingModuleCache(
  cache: QwenFullAttentionTrainingModuleCache
): void;
