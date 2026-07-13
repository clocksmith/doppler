import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export interface CausalConv1dSiluBackwardOptions {
  numTokens: number;
  channels: number;
  kernelSize: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runCausalConv1dSiluBackward(
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: CausalConv1dSiluBackwardOptions
): Promise<Tensor>;

export declare function recordCausalConv1dSiluBackward(
  recorder: CommandRecorder,
  input: Tensor,
  weight: Tensor,
  gradOutput: Tensor,
  options: CausalConv1dSiluBackwardOptions
): Promise<Tensor>;
