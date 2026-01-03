import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface Upsample2DOptions extends OutputBufferOptions {
  channels: number;
  height: number;
  width: number;
  scale?: number;
}

export interface Upsample2DSelectionContext {
  isF16: boolean;
}

export declare function runUpsample2D(
  input: Tensor,
  options: Upsample2DOptions
): Promise<Tensor>;

export declare function recordUpsample2D(
  recorder: CommandRecorder,
  input: Tensor,
  options: Upsample2DOptions
): Promise<Tensor>;
