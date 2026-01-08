/**
 * Type Casting Kernels
 *
 * Provides GPU-based type conversions:
 * - F32 to F16
 * - F16 to F32
 * - BF16 to F32
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Cast kernel options */
export interface CastOptions extends OutputBufferOptions {}

/**
 * Cast F32 buffer to F16 on GPU
 */
export declare function castF32ToF16(
  input: Tensor,
  options?: CastOptions
): Promise<Tensor>;

/**
 * Cast F16 buffer to F32 on GPU
 */
export declare function castF16ToF32(
  input: Tensor,
  options?: CastOptions
): Promise<Tensor>;

/**
 * Record F32 to F16 cast (batched, no submit)
 */
export declare function recordCastF32ToF16(
  recorder: CommandRecorder,
  input: Tensor,
  options?: CastOptions
): Promise<Tensor>;

/**
 * Record F16 to F32 cast (batched, no submit)
 */
export declare function recordCastF16ToF32(
  recorder: CommandRecorder,
  input: Tensor,
  options?: CastOptions
): Promise<Tensor>;

/**
 * Convert BF16 buffer to F32 on GPU
 */
export declare function runBF16ToF32(
  input: GPUBuffer,
  shape: readonly number[],
  name?: string
): Promise<Tensor>;

/**
 * Convert BF16 buffer to F16 on GPU (no intermediate F32 buffer)
 */
export declare function runBF16ToF16(
  input: GPUBuffer,
  shape: readonly number[],
  name?: string
): Promise<Tensor>;
