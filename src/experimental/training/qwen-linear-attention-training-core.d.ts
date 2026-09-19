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

export interface QwenLinearAttentionTrainingModuleOptions
  extends QwenLinearAttentionTrainingCoreOptions {
  hiddenSize: number;
}

export interface QwenLinearAttentionTrainingModuleInputs
  extends Omit<QwenLinearAttentionTrainingCoreInputs, 'qkv' | 'z' | 'a' | 'b'> {
  hidden: Tensor;
  qkvWeight: Tensor;
  zWeight: Tensor;
  aWeight: Tensor;
  bWeight: Tensor;
  outWeight: Tensor;
}

export interface QwenLinearAttentionTrainingModuleCache {
  dims: Record<string, number>;
  projections: Record<'qkv' | 'z' | 'a' | 'b', Tensor>;
  core: QwenLinearAttentionTrainingCoreCache;
}

export interface QwenLinearAttentionTrainingModuleForwardResult {
  output: Tensor;
  finalState: Tensor;
  cache: QwenLinearAttentionTrainingModuleCache;
}

export declare function runQwenLinearAttentionTrainingModuleForward(
  inputs: QwenLinearAttentionTrainingModuleInputs,
  options: QwenLinearAttentionTrainingModuleOptions
): Promise<QwenLinearAttentionTrainingModuleForwardResult>;

export declare function runQwenLinearAttentionTrainingModuleBackward(
  inputs: QwenLinearAttentionTrainingModuleInputs,
  gradOutput: Tensor,
  cache: QwenLinearAttentionTrainingModuleCache,
  options: QwenLinearAttentionTrainingModuleOptions
): Promise<{ hidden: Tensor; initialState: Tensor }>;

export declare function releaseQwenLinearAttentionTrainingModuleCache(
  cache: QwenLinearAttentionTrainingModuleCache
): void;
