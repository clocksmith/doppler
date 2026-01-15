import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface CrossEntropyLossOptions extends OutputBufferOptions {
  numTokens?: number;
  vocabSize?: number;
}

export declare function runCrossEntropyLoss(
  softmax: Tensor,
  targets: Tensor,
  options?: CrossEntropyLossOptions
): Promise<Tensor>;

export declare function recordCrossEntropyLoss(
  recorder: CommandRecorder,
  softmax: Tensor,
  targets: Tensor,
  options?: CrossEntropyLossOptions
): Promise<Tensor>;
