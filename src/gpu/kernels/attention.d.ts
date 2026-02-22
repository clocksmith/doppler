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
import type { KernelPathSchema } from '../../config/schema/index.js';

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
  /** Explicit kernel path context for variant selection (avoids global path state). */
  kernelPath?: KernelPathSchema | null;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Optional GPU buffer containing KV length (u32). When provided, kernel reads KV length from buffer. */
  kvLenBuffer?: GPUBuffer | null;
  /** Optional KV start offset for windowed attention (absolute position). */
  kvStart?: number;
  /** KV cache layout (contiguous, ring, paged). */
  kvLayout?: 'contiguous' | 'ring' | 'paged' | 'bdpa';
  /** Optional page table buffer for paged KV cache. */
  kvPageTable?: GPUBuffer | null;
  /** Page size for paged KV cache. */
  kvPageSize?: number;
  /** Optional indirect dispatch buffer for GPU-driven workgroup counts. */
  indirectBuffer?: GPUBuffer | null;
  /** Byte offset into indirect dispatch buffer (default: 0). */
  indirectOffset?: number;
}

export interface TieredAttentionOptions extends OutputBufferOptions {
  seqLen?: number;
  coldLen?: number;
  hotLen?: number;
  numKVHeads?: number;
  scale?: number;
  causal?: boolean;
  startPos?: number;
  attnSoftcap?: number;
  slidingWindow?: number;
  hotWindow?: number;
  hotStart?: number;
  coldPageTable?: GPUBuffer | null;
  coldPageSize?: number;
  coldLayout?: number;
  hotLayout?: number;
}

export interface TieredQuantAttentionOptions extends OutputBufferOptions {
  seqLen?: number;
  coldLen?: number;
  hotLen?: number;
  numKVHeads?: number;
  scale?: number;
  causal?: boolean;
  startPos?: number;
  attnSoftcap?: number;
  slidingWindow?: number;
  hotWindow?: number;
  hotStart?: number;
  packedStride?: number;
  mode?: 'int8' | 'int4';
}

export interface BDPAAttentionOptions extends OutputBufferOptions {
  seqLen?: number;
  kvLen?: number;
  numKVHeads?: number;
  scale?: number;
  causal?: boolean;
  startPos?: number;
  attnSoftcap?: number;
  slidingWindow?: number;
  ropeCos?: GPUBuffer | null;
  ropeSin?: GPUBuffer | null;
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
  canUseDecodeOptimized?: boolean;
  isPaged?: boolean;
  isDecode?: boolean;
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

export declare function runAttentionTiered(
  Q: Tensor,
  hotK: Tensor,
  hotV: Tensor,
  coldK: Tensor,
  coldV: Tensor,
  numHeads: number,
  headDim: number,
  options?: TieredAttentionOptions
): Promise<Tensor>;

export declare function recordAttentionTiered(
  recorder: CommandRecorder,
  Q: Tensor,
  hotK: Tensor,
  hotV: Tensor,
  coldK: Tensor,
  coldV: Tensor,
  numHeads: number,
  headDim: number,
  options?: TieredAttentionOptions
): Promise<Tensor>;

export declare function runAttentionTieredQuant(
  Q: Tensor,
  hotK: Tensor,
  hotV: Tensor,
  coldPackedK: GPUBuffer,
  coldPackedV: GPUBuffer,
  coldScalesK: GPUBuffer,
  coldScalesV: GPUBuffer,
  numHeads: number,
  headDim: number,
  options?: TieredQuantAttentionOptions
): Promise<Tensor>;

export declare function recordAttentionTieredQuant(
  recorder: CommandRecorder,
  Q: Tensor,
  hotK: Tensor,
  hotV: Tensor,
  coldPackedK: GPUBuffer,
  coldPackedV: GPUBuffer,
  coldScalesK: GPUBuffer,
  coldScalesV: GPUBuffer,
  numHeads: number,
  headDim: number,
  options?: TieredQuantAttentionOptions
): Promise<Tensor>;

export declare function runAttentionBDPA(
  Q: Tensor,
  basisK: Tensor,
  basisV: Tensor,
  pagedK: GPUBuffer,
  pagedV: GPUBuffer,
  index: GPUBuffer,
  numHeads: number,
  headDim: number,
  options?: BDPAAttentionOptions
): Promise<Tensor>;

export declare function recordAttentionBDPA(
  recorder: CommandRecorder,
  Q: Tensor,
  basisK: Tensor,
  basisV: Tensor,
  pagedK: GPUBuffer,
  pagedV: GPUBuffer,
  index: GPUBuffer,
  numHeads: number,
  headDim: number,
  options?: BDPAAttentionOptions
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
  layerIdx?: number,
  isPaged?: boolean,
  kernelPath?: KernelPathSchema | null
): {
  tier: AttentionTier;
  variant: string;
  workgroups: number;
  useF16KV: boolean;
  isDecode: boolean;
};
