import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface GroupedPointwiseConv2DOptions extends OutputBufferOptions {
  inChannels: number;
  outChannels: number;
  height: number;
  width: number;
  groups: number;
}

export declare function runGroupedPointwiseConv2D(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: GroupedPointwiseConv2DOptions
): Promise<Tensor>;

export declare function recordGroupedPointwiseConv2D(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: GroupedPointwiseConv2DOptions
): Promise<Tensor>;
