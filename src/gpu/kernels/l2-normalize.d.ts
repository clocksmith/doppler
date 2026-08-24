import type { Tensor } from '../tensor.js';

export declare function runL2Normalize(
  input: Tensor,
  options: { rowCount: number; hiddenSize: number }
): Promise<Tensor>;
