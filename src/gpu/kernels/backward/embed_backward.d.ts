import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export declare function runEmbedBackward(
  input: Tensor,
  gradOutput: Tensor,
  options?: BackwardKernelOptions
): Promise<Tensor>;

export declare function recordEmbedBackward(
  recorder: CommandRecorder,
  input: Tensor,
  gradOutput: Tensor,
  options?: BackwardKernelOptions
): Promise<Tensor>;
