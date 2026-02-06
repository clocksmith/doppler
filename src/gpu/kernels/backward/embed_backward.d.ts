import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export declare function runEmbedBackward(
  indices: Tensor,
  gradOutput: Tensor,
  options: {
    numTokens?: number;
    hiddenSize?: number;
    vocabSize?: number;
    transpose?: boolean;
    indexOffset?: number;
    outputBuffer?: GPUBuffer | null;
  }
): Promise<Tensor>;

export declare function recordEmbedBackward(
  recorder: CommandRecorder,
  indices: Tensor,
  gradOutput: Tensor,
  options: {
    numTokens?: number;
    hiddenSize?: number;
    vocabSize?: number;
    transpose?: boolean;
    indexOffset?: number;
    outputBuffer?: GPUBuffer | null;
  }
): Promise<Tensor>;
