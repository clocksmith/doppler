/**
 * Softmax Kernels
 *
 * Provides softmax operations with support for:
 * - Temperature scaling
 * - Top-K fused softmax (for MoE routing)
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Softmax kernel options */
export interface SoftmaxOptions extends OutputBufferOptions {
  batchSize?: number;
  size?: number | null;
  seqLen?: number | null;
  temperature?: number;
  normalize?: boolean;
}

export interface SoftmaxTopKOptions {
  normalize?: boolean;
  inputDtype?: 'f16' | 'f32';
  weightsDtype?: 'f16' | 'f32';
}

/**
 * Run softmax operation
 */
export declare function runSoftmax(
  input: Tensor,
  axis: number,
  options?: SoftmaxOptions
): Promise<Tensor>;

/**
 * Run fused softmax + top-K for MoE routing
 */
export declare function runSoftmaxTopK(
  logits: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options?: SoftmaxTopKOptions
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }>;

/**
 * Record softmax (batched, no submit)
 */
export declare function recordSoftmax(
  recorder: CommandRecorder,
  input: Tensor,
  axis: number,
  options?: SoftmaxOptions
): Promise<Tensor>;
