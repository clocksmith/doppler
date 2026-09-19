import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface CausalConv1dSiluOptions {
  numTokens: number;
  channels: number;
  kernelSize: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runCausalConv1dSilu(
  input: Tensor,
  weight: Tensor,
  options: CausalConv1dSiluOptions
): Promise<Tensor>;

export declare function recordCausalConv1dSilu(
  recorder: CommandRecorder,
  input: Tensor,
  weight: Tensor,
  options: CausalConv1dSiluOptions
): Promise<Tensor>;
