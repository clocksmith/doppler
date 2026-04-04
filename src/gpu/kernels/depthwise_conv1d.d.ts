import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface DepthwiseConv1DOptions extends OutputBufferOptions {
  channels: number;
  length: number;
  kernelSize: number;
}

export declare function runDepthwiseConv1D(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  options: DepthwiseConv1DOptions
): Promise<Tensor>;

export declare function recordDepthwiseConv1D(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  options: DepthwiseConv1DOptions
): Promise<Tensor>;
