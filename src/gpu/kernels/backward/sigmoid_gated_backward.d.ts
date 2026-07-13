import type { Tensor } from '../../tensor.js';

export declare function runSigmoidGatedBackward(
  input: Tensor,
  gate: Tensor,
  gradOutput: Tensor,
  options: { count: number }
): Promise<{ input: Tensor; gate: Tensor }>;

export declare function runSiluGatedBackward(
  gate: Tensor,
  up: Tensor,
  gradOutput: Tensor,
  options: { count: number; swigluLimit?: number | null }
): Promise<{ gate: Tensor; up: Tensor }>;
