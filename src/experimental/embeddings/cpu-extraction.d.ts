export declare function extractTokenEmbeddingsFromHidden(
  hiddenStates: Float32Array,
  numTokens: number,
  hiddenSize: number,
  finalNormWeights: Float32Array,
  config: Record<string, any>,
  finalNormBias?: Float32Array | null
): Float32Array;

export declare function extractEmbeddingFromHidden(
  hiddenStates: Float32Array,
  numTokens: number,
  hiddenSize: number,
  embeddingMode: 'last' | 'mean',
  finalNormWeights: Float32Array,
  config: Record<string, any>,
  embeddingPostprocessor?: Record<string, any> | null,
  normalizedTokenEmbeddings?: Float32Array | null,
  finalNormBias?: Float32Array | null
): Float32Array;
