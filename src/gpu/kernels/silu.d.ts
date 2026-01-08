/**
 * SiLU (Swish) Activation Kernels
 *
 * Provides SiLU activation with variants:
 * - Standard SiLU: x * sigmoid(x)
 * - SiLU with gating (for GLU layers)
 * - SwiGLU with row-split bias
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** SiLU kernel options */
export interface SiLUOptions extends OutputBufferOptions {
  size?: number | null;
  gate?: Tensor | null;
  useVec4?: boolean;
  biasOffset?: number;
}

/** Row-split SiLU options for fused gate+up FFN */
export interface SiLURowSplitOptions extends OutputBufferOptions {
  numTokens: number;
  dim: number;
  activation?: 'silu' | 'gelu';
}

/**
 * Run SiLU activation
 */
export declare function runSiLU(
  input: Tensor,
  options?: SiLUOptions
): Promise<Tensor>;

/**
 * Run SwiGLU with row-split bias
 */
export declare function runSwiGLURowsplitBias(
  input: Tensor,
  bias: Tensor,
  numTokens: number,
  dim: number,
  options?: SiLUOptions
): Promise<Tensor>;

/**
 * Run row-split SiLU/GELU for fused gate+up FFN.
 */
export declare function runSiLURowSplit(
  input: Tensor,
  options: SiLURowSplitOptions
): Promise<Tensor>;

/**
 * Record row-split SiLU/GELU (batched, no submit)
 */
export declare function recordSiLURowSplit(
  recorder: CommandRecorder,
  input: Tensor,
  options: SiLURowSplitOptions
): Promise<Tensor>;

/**
 * Record SiLU (batched, no submit)
 */
export declare function recordSiLU(
  recorder: CommandRecorder,
  input: Tensor,
  options?: SiLUOptions
): Promise<Tensor>;
