import type { Tensor } from '../gpu/tensor.js';

export interface AttentionBackwardOptions {
  seqLen: number;
  numHeads: number;
  headDim: number;
  scale?: number;
  causal?: boolean;
}

export interface AttentionBackwardResult {
  gradQ: Tensor;
  gradK: Tensor;
  gradV: Tensor;
}

export declare function buildAttentionSoftmaxCache(
  q: Tensor,
  k: Tensor,
  options: AttentionBackwardOptions
): Promise<Tensor>;

export declare function attentionBackwardCpu(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  softmax: Tensor | null,
  gradOutput: Tensor,
  options: AttentionBackwardOptions
): Promise<AttentionBackwardResult>;
