import type { Tensor } from '../../tensor.js';

export interface QwenGqaAttentionBackwardOptions {
  seqLen: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  scale: number;
  causal?: boolean;
}

export declare function runQwenGqaAttentionBackward(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  gradOutput: Tensor,
  options: QwenGqaAttentionBackwardOptions
): Promise<{ query: Tensor; key: Tensor; value: Tensor }>;
