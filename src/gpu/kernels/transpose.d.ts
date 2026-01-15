import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface TransposeOptions {
  outputBuffer?: GPUBuffer | null;
}

export declare function runTranspose(
  input: Tensor,
  rows: number,
  cols: number,
  options?: TransposeOptions
): Promise<Tensor>;

export declare function recordTranspose(
  recorder: CommandRecorder,
  input: Tensor,
  rows: number,
  cols: number,
  options?: TransposeOptions
): Promise<Tensor>;
