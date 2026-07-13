import type { Tensor } from '../../gpu/tensor.js';

export interface QwenLinearAttentionTrainingCoreOptions {
  numTokens: number;
  numKeyHeads: number;
  numValueHeads: number;
  keyDim: number;
  valueDim: number;
  kernelSize: number;
  checkpointInterval: number;
  queryScale: number;
  l2Eps: number;
  rmsEps: number;
}

export interface QwenLinearAttentionTrainingCoreInputs {
  qkv: Tensor;
  z: Tensor;
  a: Tensor;
  b: Tensor;
  convWeight: Tensor;
  aLog: Tensor;
  dtBias: Tensor;
  normWeight: Tensor;
  initialState: Tensor;
}

export interface QwenLinearAttentionTrainingCoreCache {
  dims: Record<string, number>;
  convolution: Tensor;
  preparation: Record<'query' | 'key' | 'value' | 'logDecay' | 'beta', Tensor>;
  recurrenceOutput: Tensor;
  checkpoints: Tensor;
}

export interface QwenLinearAttentionTrainingCoreForwardResult {
  output: Tensor;
  finalState: Tensor;
  cache: QwenLinearAttentionTrainingCoreCache;
}

export interface QwenLinearAttentionTrainingCoreGradients {
  qkv: Tensor;
  z: Tensor;
  a: Tensor;
  b: Tensor;
  initialState: Tensor;
}

export declare function runQwenLinearAttentionTrainingCoreForward(
  inputs: QwenLinearAttentionTrainingCoreInputs,
  options: QwenLinearAttentionTrainingCoreOptions
): Promise<QwenLinearAttentionTrainingCoreForwardResult>;

export declare function runQwenLinearAttentionTrainingCoreBackward(
  inputs: QwenLinearAttentionTrainingCoreInputs,
  gradOutput: Tensor,
  cache: QwenLinearAttentionTrainingCoreCache,
  options: QwenLinearAttentionTrainingCoreOptions
): Promise<QwenLinearAttentionTrainingCoreGradients>;

export declare function releaseQwenLinearAttentionTrainingCoreCache(
  cache: QwenLinearAttentionTrainingCoreCache
): void;
