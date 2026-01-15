import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';

export interface AdamOptions {
  count?: number;
  step?: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export declare function runAdam(
  params: Tensor,
  grads: Tensor,
  moment1: Tensor,
  moment2: Tensor,
  options: AdamOptions
): Promise<Tensor>;

export declare function recordAdam(
  recorder: CommandRecorder,
  params: Tensor,
  grads: Tensor,
  moment1: Tensor,
  moment2: Tensor,
  options: AdamOptions
): Promise<Tensor>;
