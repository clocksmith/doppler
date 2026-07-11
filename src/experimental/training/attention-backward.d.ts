import type { Tensor } from '../../gpu/tensor.js';

export interface AttentionBackwardOptions {
  seqLen: number;
  numHeads: number;
  numKVHeads?: number;
  headDim: number;
  scale?: number;
  causal?: boolean;
}

export interface AttentionGeometry {
  seqLen: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  headsPerKV: number;
  scale: number;
  causal: boolean;
}

export interface AttentionBackwardResult {
  gradQ: Tensor;
  gradK: Tensor;
  gradV: Tensor;
  geometry: AttentionGeometry;
}

export interface AttentionBackwardDataResult {
  dQ: Float32Array;
  dK: Float32Array;
  dV: Float32Array;
  geometry: AttentionGeometry;
}

export declare function computeAttentionSoftmaxData(
  qData: Float32Array,
  kData: Float32Array,
  options: AttentionBackwardOptions
): Float32Array;

export declare function computeAttentionBackwardData(
  qData: Float32Array,
  kData: Float32Array,
  vData: Float32Array,
  softmaxData: Float32Array,
  gradOutputData: Float32Array,
  options: AttentionBackwardOptions
): AttentionBackwardDataResult;

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
