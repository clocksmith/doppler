import type { Tensor } from '../../tensor.js';
import type { AttentionBackwardOptions, AttentionBackwardResult } from '../../../training/attention-backward.js';

export declare function runAttentionBackward(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  softmax: Tensor,
  gradOutput: Tensor,
  options?: AttentionBackwardOptions
): Promise<AttentionBackwardResult>;

export declare function recordAttentionBackward(): Promise<never>;
