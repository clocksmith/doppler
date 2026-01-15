import type { TrainingConfigSchema } from '../config/training-defaults.d.ts';
import type { Tensor } from '../gpu/tensor.js';

export declare function recordAttentionForward(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  config: TrainingConfigSchema,
  tape: unknown,
  options?: {
    seqLen?: number;
    numHeads?: number;
    numKVHeads?: number;
    headDim?: number;
    scale?: number;
    causal?: boolean;
    startPos?: number;
    attnSoftcap?: number;
    slidingWindow?: number;
    kvLen?: number;
  }
): Promise<{ output: Tensor; softmax: Tensor | null }>;
