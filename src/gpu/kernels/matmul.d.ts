/**
 * Matrix Multiplication Kernels
 *
 * Provides optimized matmul operations with support for:
 * - F16/F32 inputs and outputs
 * - Mixed precision (F16 weights, F32 activations)
 * - Tiled and naive variants
 * - Command recording for batched execution
 */

import type { Tensor, TensorDtype } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions, OutputDtypeOptions, Vec4Options } from './types.js';
import type { KernelPathSchema } from '../../config/schema/index.js';

/** Matmul kernel options */
export interface MatmulOptions extends OutputBufferOptions, OutputDtypeOptions, Vec4Options {
  alpha?: number;
  /** Optional matmul role for kernel path overrides (e.g., 'q_proj', 'ffn_gate', 'lm_head') */
  role?: string;
  /** Layer index for kernel path layer overrides */
  layerIdx?: number;
  /** Explicit kernel path context for variant selection (avoids global path state). */
  kernelPath?: KernelPathSchema | null;
  /**
   * Whether B matrix is stored transposed.
   * - true: B is [N,K] (SafeTensors/row-major), needs transpose
   * - false: B is [K,N] (column-major/pre-transposed), direct access
   * - 'auto': Auto-detect from buffer layout metadata (default)
   */
  transposeB?: boolean | 'auto';
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  aDtype?: 'f16' | 'f32' | null;
  bDtype?: 'f16' | 'f32' | 'q4k' | null;
  preferF16?: boolean;
  /** WGSL override constants for pipeline creation */
  constants?: Record<string, number | boolean>;
}

/** Context for base matmul kernel selection rules. */
export interface MatmulKernelSelectionContext {
  useF16Matmul: boolean;
  useF16wF32a: boolean;
  useVec4: boolean;
}

/** Context for fused Q4K variant selection rules. */
export interface MatmulQ4KFusedContext {
  useF16A: boolean;
  useF16Out: boolean;
  isM1: boolean;
}

/** Context for GEMV variant selection rules. */
export interface GemvSelectionContext {
  hasSubgroups: boolean;
  useF16Gemv: boolean;
  useF32Gemv: boolean;
  useMulticol: boolean;
}

/** Context for top-level matmul variant selection rules. */
export interface MatmulVariantSelectionContext {
  canFused: boolean;
  useGemv: boolean;
}

/**
 * Check if fused Q4K kernels are disabled.
 */
export declare function isFusedQ4KDisabled(): boolean;

/**
 * Select the best matmul kernel variant
 */
export declare function selectMatmulKernel(options?: MatmulOptions): string;

/**
 * Create bind group layout for matmul operation
 */
export declare function createMatmulBindGroupLayout(): GPUBindGroupLayout;

/**
 * Run matrix multiplication
 */
export declare function runMatmul(
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options?: MatmulOptions
): Promise<Tensor>;

/**
 * Record matrix multiplication (batched, no submit)
 */
export declare function recordMatmul(
  recorder: CommandRecorder,
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options?: MatmulOptions
): Promise<Tensor>;
