import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface GatedRmsNormOptions {
  rows: number;
  width: number;
  eps: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runGatedRmsNorm(
  input: Tensor,
  gate: Tensor,
  weight: Tensor,
  options: GatedRmsNormOptions
): Promise<Tensor>;

export declare function recordGatedRmsNorm(
  recorder: CommandRecorder,
  input: Tensor,
  gate: Tensor,
  weight: Tensor,
  options: GatedRmsNormOptions
): Promise<Tensor>;
