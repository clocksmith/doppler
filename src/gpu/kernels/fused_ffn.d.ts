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
  /** WGSL override constants resolved from the active kernel path. */
  pipelineConstants?: Record<string, number | boolean> | null;
  /** Explicit config-owned Q4_K decode kernel variant. */
  variant?: 'q4k_metal_simd16' | null;
}

export interface FusedNormedFFNOptions extends FusedFFNOptions {
  /** Use RMSNorm offset semantics: output = x * invRms * (1 + weight). */
  rmsNormWeightOffset?: boolean;
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

export declare function runFusedFFNFromRMSNormStats(
  input: Tensor,
  invRmsBuffer: GPUBuffer,
  normWeight: GPUBuffer | WeightBuffer,
  W_gate: GPUBuffer | WeightBuffer,
  W_up: GPUBuffer | WeightBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options?: FusedNormedFFNOptions
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

export declare function recordFusedFFNFromRMSNormStats(
  recorder: CommandRecorder,
  input: Tensor,
  invRmsBuffer: GPUBuffer,
  normWeight: GPUBuffer | WeightBuffer,
  W_gate: GPUBuffer | WeightBuffer,
  W_up: GPUBuffer | WeightBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options?: FusedNormedFFNOptions
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
