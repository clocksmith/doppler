import type { CommandRecorder } from '../../command-recorder.js';
import type { Tensor } from '../../tensor.js';
import type { BackwardKernelOptions } from './utils.js';

export interface RoPEBackwardOptions extends BackwardKernelOptions {
  seqLen: number;
  numHeads: number;
  headDim: number;
  startPos?: number;
}

export declare function runRoPEBackward(
  gradOutput: Tensor,
  freqsCos: Tensor,
  freqsSin: Tensor,
  options: RoPEBackwardOptions
): Promise<Tensor>;

export declare function recordRoPEBackward(
  recorder: CommandRecorder,
  gradOutput: Tensor,
  freqsCos: Tensor,
  freqsSin: Tensor,
  options: RoPEBackwardOptions
): Promise<Tensor>;
