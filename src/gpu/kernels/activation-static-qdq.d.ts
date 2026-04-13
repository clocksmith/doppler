import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface ActivationStaticQdqOptions {
  count?: number;
  outputBuffer?: GPUBuffer | null;
  qmin?: number;
  qmax?: number;
}

export declare function runActivationStaticQdq(
  input: Tensor,
  scale: number,
  options?: ActivationStaticQdqOptions
): Promise<Tensor>;

export declare function recordActivationStaticQdq(
  recorder: CommandRecorder,
  input: Tensor,
  scale: number,
  options?: ActivationStaticQdqOptions
): Promise<Tensor>;
