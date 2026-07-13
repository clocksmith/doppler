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

export interface QwenFullAttentionLoraAdapter {
  A: Tensor;
  B: Tensor;
  rank: number;
  alpha: number;
}

export interface QwenFullAttentionLoraAdapters {
  q?: QwenFullAttentionLoraAdapter;
  k?: QwenFullAttentionLoraAdapter;
  v?: QwenFullAttentionLoraAdapter;
  o?: QwenFullAttentionLoraAdapter;
}

export interface QwenFullAttentionLoraGradients {
  q: { A: Tensor | null; B: Tensor | null };
  k: { A: Tensor | null; B: Tensor | null };
  v: { A: Tensor | null; B: Tensor | null };
  o: { A: Tensor | null; B: Tensor | null };
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
  lora?: QwenFullAttentionLoraAdapters;
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
  gated: Tensor;
  projectionDowns: {
    q: Tensor | null;
    k: Tensor | null;
    v: Tensor | null;
    o: Tensor | null;
  };
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
): Promise<{ hidden: Tensor; lora: QwenFullAttentionLoraGradients }>;

export declare function releaseQwenFullAttentionTrainingModuleCache(
  cache: QwenFullAttentionTrainingModuleCache
): void;
