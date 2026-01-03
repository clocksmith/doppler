/**
 * Residual Connection Kernels
 *
 * Provides element-wise addition operations for:
 * - Residual connections (add two tensors)
 * - Bias addition
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Residual kernel options */
export interface ResidualOptions extends OutputBufferOptions {
  useVec4?: boolean;
  dataOffset?: number;
  biasOffset?: number;
}

/** Context for residual variant selection rules. */
export interface ResidualSelectionContext {
  outputDtype: 'f16' | 'f32';
  useVec4: boolean;
}

/** Context for bias add variant selection rules. */
export interface BiasAddSelectionContext {
  dataDtype: 'f16' | 'f32';
  biasDtype: 'f16' | 'f32';
}

/**
 * Run residual add (element-wise addition)
 */
export declare function runResidualAdd(
  a: Tensor,
  b: Tensor,
  size: number,
  options?: ResidualOptions
): Promise<Tensor>;

/**
 * Run bias add
 */
export declare function runBiasAdd(
  data: Tensor,
  bias: Tensor,
  numTokens: number,
  dim: number,
  options?: ResidualOptions
): Promise<Tensor>;

/**
 * Record residual add (batched, no submit)
 */
export declare function recordResidualAdd(
  recorder: CommandRecorder,
  a: Tensor,
  b: Tensor,
  size: number,
  options?: ResidualOptions
): Promise<Tensor>;

/**
 * Record bias add (batched, no submit)
 */
export declare function recordBiasAdd(
  recorder: CommandRecorder,
  data: Tensor,
  bias: Tensor,
  numTokens: number,
  dim: number,
  options?: ResidualOptions
): Promise<Tensor>;
