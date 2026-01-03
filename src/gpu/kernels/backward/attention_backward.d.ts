import type { Tensor } from '../../tensor.js';
import type { AttentionBackwardOptions, AttentionBackwardResult } from '../../../training/attention-backward.js';
import type { CommandRecorder } from '../../command-recorder.js';

export declare function runAttentionBackward(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  softmax: Tensor,
  gradOutput: Tensor,
  options?: AttentionBackwardOptions
): Promise<AttentionBackwardResult>;

export declare function recordAttentionBackward(
  recorder: CommandRecorder,
  q: Tensor,
  k: Tensor,
  v: Tensor,
  softmax: Tensor,
  gradOutput: Tensor,
  options?: AttentionBackwardOptions
): Promise<AttentionBackwardResult>;
