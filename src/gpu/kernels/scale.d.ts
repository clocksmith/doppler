/**
 * Scale kernel - multiply each element by a scalar factor
 * Used for embedding scaling in Gemma models (sqrt(hidden_size))
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Scale kernel options */
export interface ScaleOptions extends OutputBufferOptions {
  /** Number of elements to scale (inferred from buffer size if not provided) */
  count?: number;
  /** Whether to scale in-place (output = input) */
  inplace?: boolean;
}

/**
 * Run scale operation: output = input * scale
 */
export declare function runScale(
  input: Tensor,
  scale: number,
  options?: ScaleOptions
): Promise<Tensor>;

/**
 * Record scale operation (batched, no submit)
 */
export declare function recordScale(
  recorder: CommandRecorder,
  input: Tensor,
  scale: number,
  options?: ScaleOptions
): Promise<Tensor>;
