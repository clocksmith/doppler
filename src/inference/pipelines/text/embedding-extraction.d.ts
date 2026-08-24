import type { WeightBuffer } from '../../../gpu/weight-buffer.js';

export interface EmbeddingExtractionOptions {
  hiddenBuffer: GPUBuffer;
  activationDtype: 'f16' | 'f32';
  numTokens: number;
  hiddenSize: number;
  embeddingMode: 'last' | 'mean';
  finalNorm: GPUBuffer | WeightBuffer | ArrayBufferView;
  finalNormBias?: Float32Array | null;
  config: Record<string, any>;
  embeddingPostprocessor?: Record<string, any> | null;
  returnTokenEmbeddings?: boolean;
  sequencePooling?: { mode: 'last' | 'mean'; excludeTokenIds: Iterable<number> } | null;
  tokenIds?: ArrayLike<number> | null;
}

export interface EmbeddingExtractionResult {
  embedding: Float32Array;
  tokenEmbeddings: Float32Array | null;
  pooledSequenceEmbedding: Float32Array | null;
  tokenMask: Uint8Array | null;
  includedTokenCount: number;
}

export declare function extractEmbeddingFromHiddenGPU(
  options: EmbeddingExtractionOptions
): Promise<EmbeddingExtractionResult>;
