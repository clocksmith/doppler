/**
 * Fused GEMV + RMSNorm Kernel
 *
 * For decode (M=1), combines the down projection matmul with RMSNorm in a single kernel.
 */

import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Fused MatmulRMSNorm kernel options */
export interface MatmulRMSNormFusedOptions extends OutputBufferOptions {
  /** Output dimension N (hiddenSize) */
  N: number;
  /** Input dimension K (intermediateSize) */
  K: number;
  /** RMSNorm epsilon (default: 1e-5) */
  eps?: number;
  /** Optional residual buffer to add to output */
  residual?: GPUBuffer | null;
  /**
   * Whether weight matrix is stored transposed.
   * - true: weight is [N,K] (row-major/SafeTensors), needs transpose access
   * - false: weight is [K,N] (column-major/pre-transposed), direct access
   * Default: true (matches GGUF convention)
   */
  transposeB?: boolean;
}

/**
 * Select fused kernel variant based on output size
 */
export declare function selectMatmulRMSNormFusedVariant(N: number): string;

/**
 * Run fused GEMV + RMSNorm
 */
export declare function runMatmulRMSNormFused(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<Tensor>;

/**
 * Record fused GEMV + RMSNorm (batched, no submit)
 */
export declare function recordMatmulRMSNormFused(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<Tensor>;

/**
 * Check if fused kernel should be used for given dimensions
 */
export declare function shouldUseFusedMatmulRMSNorm(M: number, N: number): boolean;
