import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface Conv2DOptions extends OutputBufferOptions {
  inChannels: number;
  outChannels: number;
  height: number;
  width: number;
  kernelH: number;
  kernelW: number;
  stride?: number;
  pad?: number;
}

export interface Conv2DSelectionContext {
  isF16: boolean;
}

export declare function runConv2D(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: Conv2DOptions
): Promise<Tensor>;

export declare function recordConv2D(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  bias: GPUBuffer | WeightBuffer | null,
  options: Conv2DOptions
): Promise<Tensor>;
