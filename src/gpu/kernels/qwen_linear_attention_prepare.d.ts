import type { Tensor } from '../tensor.js';

export interface QwenLinearAttentionPrepareOptions {
  numTokens: number;
  numKeyHeads: number;
  numValueHeads: number;
  keyDim: number;
  valueDim: number;
  eps: number;
}

export interface QwenLinearAttentionPrepared {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  logDecay: Tensor;
  beta: Tensor;
}

export declare function runQwenLinearAttentionPrepare(
  mixed: Tensor,
  a: Tensor,
  b: Tensor,
  aLog: Tensor,
  dtBias: Tensor,
  options: QwenLinearAttentionPrepareOptions
): Promise<QwenLinearAttentionPrepared>;
