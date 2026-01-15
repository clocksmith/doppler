import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export declare function runSoftmaxBackward(
  input: Tensor,
  gradOutput: Tensor,
  options: BackwardKernelOptions & { rows: number; cols: number }
): Promise<Tensor>;

export declare function recordSoftmaxBackward(
  recorder: CommandRecorder,
  input: Tensor,
  gradOutput: Tensor,
  options: BackwardKernelOptions & { rows: number; cols: number }
): Promise<Tensor>;
