export interface VisionAttentionParams {
  qkv: GPUBuffer;
  seqLen: number;
  numHeads: number;
  headDim: number;
  hiddenSize: number;
}

export declare function computeVisionAttention(params: VisionAttentionParams): Promise<GPUBuffer>;
