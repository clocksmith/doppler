import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface GroupNormOptions extends OutputBufferOptions {
  channels: number;
  height: number;
  width: number;
  numGroups: number;
  eps: number;
}

export interface GroupNormSelectionContext {
  isF16: boolean;
}

export declare function runGroupNorm(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer,
  options: GroupNormOptions
): Promise<Tensor>;

export declare function recordGroupNorm(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer,
  options: GroupNormOptions
): Promise<Tensor>;
