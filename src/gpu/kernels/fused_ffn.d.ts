/**
 * Fused FFN Kernel (Tier 2 P0)
 *
 * EXPERIMENTAL: Not currently wired into layer.ts.
 * Complete gate+up fusion kernel, kept for future integration.
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { OutputBufferOptions } from './types.js';

/** FFN activation type */
export type FFNActivation = 'silu' | 'gelu';

/** Fused FFN options */
export interface FusedFFNOptions extends OutputBufferOptions {
  /** Batch size (default: 1) */
  batchSize?: number;
  /** Activation function (default: 'silu') */
  activation?: FFNActivation;
  /** Scale factor (default: 1.0) */
  alpha?: number;
  /** Clamp SwiGLU output (null = disabled) */
  swigluLimit: number | null;
}

/**
 * Run fused FFN forward pass
 */
export declare function runFusedFFN(
  input: Tensor,
  W_gate: GPUBuffer | WeightBuffer,
  W_up: GPUBuffer | WeightBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options?: FusedFFNOptions
): Promise<Tensor>;

/**
 * Record fused FFN forward pass (batched, no submit)
 */
export declare function recordFusedFFN(
  recorder: CommandRecorder,
  input: Tensor,
  W_gate: GPUBuffer | WeightBuffer,
  W_up: GPUBuffer | WeightBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options?: FusedFFNOptions
): Promise<Tensor>;

/**
 * Calculate memory savings from using fused FFN
 */
export declare function calculateFusedFFNSavings(
  batchSize: number,
  hiddenSize: number,
  intermediateSize: number
): {
  separateBytes: number;
  fusedBytes: number;
  savingsBytes: number;
  savingsPct: number;
};
