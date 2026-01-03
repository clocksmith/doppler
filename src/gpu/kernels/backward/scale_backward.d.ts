import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export declare function runScaleBackward(
  input: Tensor,
  gradOutput: Tensor,
  options: BackwardKernelOptions & { scale: number }
): Promise<Tensor>;

export declare function recordScaleBackward(
  recorder: CommandRecorder,
  input: Tensor,
  gradOutput: Tensor,
  options: BackwardKernelOptions & { scale: number }
): Promise<Tensor>;
