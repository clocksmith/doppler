/**
 * Fused GEMV + Residual Kernel
 *
 * For decode (M=1), combines output projection matmul with residual add in a single kernel.
 */

import type { Tensor, TensorDtype } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Fused MatmulResidual kernel options */
export interface MatmulResidualFusedOptions extends OutputBufferOptions {
  /** Output dimension N (hiddenSize) */
  N: number;
  /** Input dimension K (numHeads * headDim) */
  K: number;
  /** Scaling factor (default: 1.0) */
  alpha?: number;
}

/**
 * Check if fused GEMV+residual should be used.
 */
export declare function shouldUseFusedMatmulResidual(M: number): boolean;

/**
 * Run fused GEMV + Residual
 */
export declare function runMatmulResidualFused(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  residual: Tensor,
  options: MatmulResidualFusedOptions
): Promise<Tensor>;

/**
 * Record fused GEMV + Residual (batched, no submit)
 */
export declare function recordMatmulResidualFused(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  residual: Tensor,
  options: MatmulResidualFusedOptions
): Promise<Tensor>;
