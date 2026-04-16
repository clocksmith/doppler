import type { Tensor } from '../../gpu/tensor.js';

export interface TokenizedSample {
  inputIds: number[];
  targetIds: number[];
  text?: string;
}

export interface TokenBatch {
  inputFlat: Uint32Array;
  targetFlat: Uint32Array;
  offsets: number[];
}

export declare function buildTokenBatch(samples: TokenizedSample[]): TokenBatch;

export declare function createTokenBatchTensors(batch: TokenBatch): {
  input: Tensor;
  targets: Tensor;
  offsets: number[];
};
