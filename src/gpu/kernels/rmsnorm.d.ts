/**
 * RMSNorm Kernels
 *
 * Provides RMS normalization with optional residual connection.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { WeightBuffer, TensorLike } from '../weight-buffer.js';

/** Maximum hidden size supported by the cached residual RMSNorm variant. */
export declare const RMSNORM_CACHE_LIMIT: number;

/** RMSNorm kernel options */
export interface RMSNormOptions extends OutputBufferOptions {
  batchSize?: number;
  hiddenSize?: number | null;
  residual?: Tensor | null;
  preResidual?: Tensor | null;
  residualSumOutput?: GPUBuffer | Tensor | null;
  outputScale?: number | null;
  /** Use (1+w)*x normalization for Gemma 2/3 */
  rmsNormWeightOffset?: boolean;
  label?: string;
}

/**
 * Return true when the residual RMSNorm variant must avoid the cached WGSL path.
 */
export declare function residualVariantBypassesCache(
  residual: Tensor | GPUBuffer | WeightBuffer | TensorLike | null | undefined,
  hiddenSize: number | null | undefined
): boolean;

export declare function resolveNormWeightDtype(
  weight: GPUBuffer | WeightBuffer | TensorLike,
  hiddenSize: number | null | undefined
): string;

export declare function assertRMSNormWeightBuffer(
  weight: GPUBuffer | WeightBuffer | TensorLike,
  weightBuffer: GPUBuffer | null | undefined,
  hiddenSize: number | null | undefined
): void;

export declare function planRMSNormDispatch(
  target: CommandRecorder | null | undefined,
  numTokens: number
): { tokenStride: number; workgroups: [number, number, number] };

/**
 * Select RMSNorm kernel variant based on options, tensor dtypes, and GPU capabilities.
 */
export declare function selectRMSNormKernel(options?: RMSNormOptions, isF16?: boolean): string;

/**
 * Run RMSNorm
 */
export declare function runRMSNorm(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options?: RMSNormOptions
): Promise<Tensor>;

/**
 * Record RMSNorm (batched, no submit)
 */
export declare function recordRMSNorm(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer | TensorLike,
  eps: number,
  options?: RMSNormOptions
): Promise<Tensor>;
