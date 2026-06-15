import type { Tensor } from '../tensor.js';
import type { SplitWeightBuffer, WeightBuffer } from '../weight-buffer.js';
import type { OutputBufferOptions } from './types.js';

export interface SoftEmbeddingSplitOptions extends OutputBufferOptions {}

export interface SoftEmbeddingLogitsOptions {
  temperature?: number;
  chunkRows?: number;
}

export declare function runSoftEmbeddingSplitF16(
  softmaxTensor: Tensor,
  splitEmbedding: SplitWeightBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options?: SoftEmbeddingSplitOptions
): Promise<Tensor>;

export declare function runSoftEmbeddingLogitsF16(
  logitsTensor: Tensor,
  embedding: WeightBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options?: SoftEmbeddingLogitsOptions
): Promise<Tensor>;
