import type { Tensor } from '../tensor.js';

export interface EmbeddingPoolOptions {
  rowCount: number;
  hiddenSize: number;
  mode: 'mean' | 'last';
  mask?: Uint32Array | null;
  includedCount?: number;
  lastIndex?: number;
}

export declare function runEmbeddingPool(
  input: Tensor,
  options: EmbeddingPoolOptions
): Promise<Tensor>;
