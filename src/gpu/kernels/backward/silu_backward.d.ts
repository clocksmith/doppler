import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './executor.js';

export declare function runSiluBackward(
  input: Tensor,
  gradOutput: Tensor,
  options?: BackwardKernelOptions
): Promise<Tensor>;

export declare function recordSiluBackward(
  recorder: CommandRecorder,
  input: Tensor,
  gradOutput: Tensor,
  options?: BackwardKernelOptions
): Promise<Tensor>;
