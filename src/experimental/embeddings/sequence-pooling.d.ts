export interface SequenceEmbeddingPoolResult {
  pooled: Float32Array;
  tokenMask: Uint8Array;
  includedTokenCount: number;
}

export declare function poolSequenceTokenEmbeddings(
  tokenEmbeddings: Float32Array,
  tokenIds: ArrayLike<number>,
  hiddenSize: number,
  pooling: { mode: 'last' | 'mean'; excludeTokenIds: Iterable<number> }
): SequenceEmbeddingPoolResult;
