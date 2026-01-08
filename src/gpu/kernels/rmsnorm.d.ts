/**
 * RMSNorm Kernels
 *
 * Provides RMS normalization with optional residual connection.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** RMSNorm kernel options */
export interface RMSNormOptions extends OutputBufferOptions {
  batchSize?: number;
  hiddenSize?: number | null;
  residual?: Tensor | null;
}

/**
 * Select RMSNorm kernel variant based on options, tensor dtypes, and GPU capabilities.
 */
export declare function selectRMSNormKernel(options?: RMSNormOptions, isF16?: boolean): string;

/**
 * Run RMSNorm
 */
export declare function runRMSNorm(
  input: Tensor,
  weight: GPUBuffer,
  eps?: number,
  options?: RMSNormOptions
): Promise<Tensor>;

/**
 * Record RMSNorm (batched, no submit)
 */
export declare function recordRMSNorm(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer,
  eps?: number,
  options?: RMSNormOptions
): Promise<Tensor>;
