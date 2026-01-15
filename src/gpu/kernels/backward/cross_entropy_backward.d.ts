import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export interface CrossEntropyBackwardOptions extends BackwardKernelOptions {
  numTokens: number;
  vocabSize: number;
}

export declare function runCrossEntropyBackward(
  softmax: Tensor,
  targets: Tensor,
  options: CrossEntropyBackwardOptions
): Promise<Tensor>;

export declare function recordCrossEntropyBackward(
  recorder: CommandRecorder,
  softmax: Tensor,
  targets: Tensor,
  options: CrossEntropyBackwardOptions
): Promise<Tensor>;
