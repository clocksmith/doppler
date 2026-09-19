import type { Tensor } from '../tensor.js';

export interface QwenAttentionSplitQGateOptions {
  numTokens: number;
  numHeads: number;
  headDim: number;
}

export declare function runQwenAttentionSplitQGate(
  input: Tensor,
  options: QwenAttentionSplitQGateOptions
): Promise<{ query: Tensor; gate: Tensor }>;
