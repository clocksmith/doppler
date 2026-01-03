/**
 * RoPE (Rotary Position Embedding) Kernels
 *
 * Provides rotary position embedding with multiple variants:
 * - Standard RoPE
 * - NTK-scaled RoPE
 * - YaRN (Yet another RoPE extensioN)
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** RoPE kernel options */
export interface RoPEOptions extends OutputBufferOptions {
  numHeads?: number;
  headDim?: number;
  ropeTheta?: number;
  startPos?: number;
}

/** Context for RoPE variant selection rules. */
export interface RoPESelectionContext {
  useF16: boolean;
}

/**
 * Run RoPE operation
 */
export declare function runRoPE(
  input: Tensor,
  freqsCos: GPUBuffer,
  freqsSin: GPUBuffer,
  seqLen: number,
  options?: RoPEOptions
): Promise<Tensor>;

/**
 * Record RoPE (batched, no submit)
 */
export declare function recordRoPE(
  recorder: CommandRecorder,
  input: Tensor,
  freqsCos: GPUBuffer,
  freqsSin: GPUBuffer,
  seqLen: number,
  options?: RoPEOptions
): Promise<Tensor>;
