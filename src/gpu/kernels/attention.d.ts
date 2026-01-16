/**
 * Attention Kernels
 *
 * Provides optimized attention operations with support for:
 * - Prefill and decode phases
 * - Causal masking
 * - Grouped-query attention (GQA)
 * - Multiple implementation tiers (tiled, streaming)
 * - F16/F32 KV cache support
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

/** Attention kernel options */
export interface AttentionOptions extends OutputBufferOptions {
  seqLen?: number;
  kvLen?: number;
  numKVHeads?: number;
  scale?: number;
  causal?: boolean;
  startPos?: number;
  slidingWindow?: number;
  /** Layer index for kernel path layer overrides */
  layerIdx?: number;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Optional GPU buffer containing KV length (u32). When provided, kernel reads KV length from buffer. */
  kvLenBuffer?: GPUBuffer | null;
  /** Optional indirect dispatch buffer for GPU-driven workgroup counts. */
  indirectBuffer?: GPUBuffer | null;
  /** Byte offset into indirect dispatch buffer (default: 0). */
  indirectOffset?: number;
}

export type AttentionTier = 'subgroup' | 'tiled_large' | 'tiled_small' | 'streaming';

/** Context for attention tier selection rules. */
export interface AttentionTierContext {
  canSubgroup: boolean;
  canLarge: boolean;
  canSmall: boolean;
  isDecode: boolean;
}

/** Context for attention variant selection rules. */
export interface AttentionVariantContext {
  tier: AttentionTier;
  useF16KV: boolean;
  canUseChunked: boolean;
  canUseDecodeSubgroup: boolean;
}

/**
 * Run attention operation
 */
export declare function runAttention(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  mask: GPUBuffer | null,
  numHeads: number,
  headDim: number,
  options?: AttentionOptions
): Promise<Tensor>;

/**
 * Record attention operation (batched, no submit)
 */
export declare function recordAttention(
  recorder: CommandRecorder,
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  mask: GPUBuffer | null,
  numHeads: number,
  headDim: number,
  options?: AttentionOptions
): Promise<Tensor>;

export declare function resolveAttentionPlanForTest(
  seqLen: number,
  kvLen: number,
  headDim: number,
  numHeads: number,
  kvDtype: 'f16' | 'f32',
  qDtype: 'f16' | 'f32',
  sharedLimit: number,
  caps: { hasSubgroups: boolean; hasF16?: boolean },
  layerIdx?: number
): {
  tier: AttentionTier;
  variant: string;
  workgroups: number;
  useF16KV: boolean;
  isDecode: boolean;
};
