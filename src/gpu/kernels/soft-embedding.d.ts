import type { Tensor } from '../tensor.js';
import type { SplitWeightBuffer } from '../weight-buffer.js';
import type { OutputBufferOptions } from './types.js';

export interface SoftEmbeddingSplitOptions extends OutputBufferOptions {}

export declare function runSoftEmbeddingSplitF16(
  softmaxTensor: Tensor,
  splitEmbedding: SplitWeightBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  options?: SoftEmbeddingSplitOptions
): Promise<Tensor>;
