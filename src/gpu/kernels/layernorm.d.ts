/**
 * LayerNorm Kernels
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface LayerNormOptions extends OutputBufferOptions {
  batchSize?: number;
  hiddenSize?: number | null;
  /** Manifest/materialization-owned dtype for both weight and bias buffers. */
  normWeightDtype: 'f16' | 'f32';
}

export declare function selectLayerNormKernel(options?: LayerNormOptions, isF16?: boolean): string;

export declare function runLayerNorm(
  input: Tensor,
  weight: GPUBuffer,
  bias: GPUBuffer,
  eps: number,
  options: LayerNormOptions
): Promise<Tensor>;

export declare function recordLayerNorm(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer,
  bias: GPUBuffer,
  eps: number,
  options: LayerNormOptions
): Promise<Tensor>;
