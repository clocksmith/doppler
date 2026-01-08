/**
 * GeLU Activation Kernels
 *
 * Provides GeLU activation: x * Phi(x) where Phi is the CDF of standard normal distribution.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** GeLU kernel options */
export interface GeLUOptions extends OutputBufferOptions {
  size?: number | null;
  gate?: Tensor | null;
}

/**
 * Run GeLU activation
 */
export declare function runGeLU(
  input: Tensor,
  options?: GeLUOptions
): Promise<Tensor>;

/**
 * Record GeLU (batched, no submit)
 * Supports gated variant (GeGLU) when options.gate is provided.
 */
export declare function recordGeLU(
  recorder: CommandRecorder,
  input: Tensor,
  options?: GeLUOptions
): Promise<Tensor>;
