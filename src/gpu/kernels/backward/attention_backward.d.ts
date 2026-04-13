import type { Tensor } from '../../tensor.js';
import type { CommandRecorder } from '../../command-recorder.js';

export interface AttentionBackwardOptions {
  seqLen: number;
  numHeads: number;
  headDim: number;
  scale?: number;
  causal?: boolean;
}

export interface AttentionBackwardResult {
  gradQ: Tensor;
  gradK: Tensor;
  gradV: Tensor;
}

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
