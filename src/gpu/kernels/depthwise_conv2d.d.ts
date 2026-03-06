import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface DepthwiseConv2DOptions extends OutputBufferOptions {
  channels: number;
  height: number;
  width: number;
  kernelH: number;
  kernelW: number;
  stride?: number;
  pad?: number;
}

export declare function runDepthwiseConv2D(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: DepthwiseConv2DOptions
): Promise<Tensor>;

export declare function recordDepthwiseConv2D(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: DepthwiseConv2DOptions
): Promise<Tensor>;
