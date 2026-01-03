import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export interface RmsNormBackwardOptions extends BackwardKernelOptions {
  numTokens: number;
  hiddenSize: number;
  eps?: number;
}

export declare function runRmsNormBackward(
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: RmsNormBackwardOptions
): Promise<Tensor>;

export declare function recordRmsNormBackward(
  recorder: CommandRecorder,
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: RmsNormBackwardOptions
): Promise<Tensor>;
