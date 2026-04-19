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
import type { ExecutionV1PoliciesSchema } from '../../config/schema/execution-v1.schema.js';

/** RoPE kernel options */
export interface RoPEOptions extends OutputBufferOptions {
  numHeads?: number;
  headDim?: number;
  rotaryDim?: number;
  interleaved?: boolean;
  ropeTheta?: number;
  startPos?: number;
  executionPolicies?: ExecutionV1PoliciesSchema | null;
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
