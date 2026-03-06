import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface ReLUOptions extends OutputBufferOptions {
  count?: number | null;
}

export declare function runReLU(
  input: Tensor,
  options?: ReLUOptions
): Promise<Tensor>;

export declare function recordReLU(
  recorder: CommandRecorder,
  input: Tensor,
  options?: ReLUOptions
): Promise<Tensor>;
