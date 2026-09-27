/**
 * Weight buffer management utilities.
 *
 * This module handles:
 * - Creating GPU buffers from CPU weight data
 * - Handling RMSNorm weight buffers (offset is applied at runtime)
 * - Type guards for layer weight structures
 * - Buffer lifecycle management
 *
 * @module inference/pipelines/text/weights
 */

import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { Tensor } from '../../../gpu/tensor.js';

/**
 * Weight type that can be a raw GPUBuffer, a typed WeightBuffer, or CPU Float32Array.
 * WeightBuffer provides explicit dtype/layout metadata; GPUBuffer uses WeakMap tracking.
 */
export type LayerWeightBuffer = GPUBuffer | WeightBuffer | Float32Array | CpuWeightBuffer;

/**
 * Weights for a single transformer layer.
 */
export interface LayerWeights {
  // Attention
  inputNorm: GPUBuffer | Float32Array;
  inputNormBias?: GPUBuffer | Float32Array | null;
  qProj: LayerWeightBuffer;
  qProjBias?: GPUBuffer | Float32Array | null;
  kProj: LayerWeightBuffer;
  kProjBias?: GPUBuffer | Float32Array | null;
  vProj?: LayerWeightBuffer;
  vProjBias?: GPUBuffer | Float32Array | null;
  oProj: LayerWeightBuffer;
  oProjBias?: GPUBuffer | Float32Array | null;
  qGateProj?: LayerWeightBuffer | null;
  convInProj?: LayerWeightBuffer;
  convKernel?: LayerWeightBuffer;
  convOutProj?: LayerWeightBuffer;
  /** Fused Q/K/V projection (runtime-generated for 3->1 matmul optimization) */
  qkvProj?: GPUBuffer | WeightBuffer | null;
  /** Concatenated Q/K/V projection bias matching qkvSizes. */
  qkvProjBias?: GPUBuffer | Float32Array | null;
  /** Sizes for splitting fused QKV output: [qSize, kSize, vSize] in elements */
  qkvSizes?: [number, number, number];
  /** Data type of fused QKV weights (f16, f32, or q4k) */
  qkvDtype?: 'f16' | 'f32' | 'q4k';
  /** Fused linear-attention A/B projection weight for decode-only A+B projection fusion. */
  linearABProj?: WeightBuffer | null;
  /** Fused linear-attention QKV/Z projection weight for decode-only QKV+Z projection fusion. */
  linearQKVZProj?: WeightBuffer | null;

  // FFN (dense layers)
  postAttentionNorm?: GPUBuffer | Float32Array;
  postAttentionNormBias?: GPUBuffer | Float32Array | null;
  postAttnNorm?: GPUBuffer | Float32Array;  // LLaMA-style pre-FFN norm
  gate?: LayerWeightBuffer;
  up?: LayerWeightBuffer;
  down?: LayerWeightBuffer;
  gateUp?: LayerWeightBuffer;  // Fused gate+up for 2-pass FFN

  // Feed-forward normalization
  preFeedforwardNorm?: GPUBuffer | Float32Array;
  preFeedforwardNormBias?: GPUBuffer | Float32Array | null;
  preFeedforwardNorm2?: GPUBuffer | Float32Array;
  postFeedforwardNorm?: GPUBuffer | Float32Array;
  postFeedforwardNormBias?: GPUBuffer | Float32Array | null;
  postFeedforwardNorm1?: GPUBuffer | Float32Array;
  postFeedforwardNorm2?: GPUBuffer | Float32Array;
  layerScalar?: Float32Array | null;

  // MoE
  routerWeight?: GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer | Float32Array;
  routerBias?: GPUBuffer | Float32Array | null;
  routerScale?: GPUBuffer | Float32Array | null;
  routerPerExpertScale?: GPUBuffer | Float32Array | null;
  qNorm?: GPUBuffer | Float32Array;
  kNorm?: GPUBuffer | Float32Array;
  experts?: ExpertWeights[];
}

/**
 * Weights for a single MoE expert.
 */
export interface ExpertWeights {
  expertFormat?: 'mixtral' | 'gpt-oss' | 'gemma4';
  gate?: LayerWeightBuffer;
  up?: LayerWeightBuffer;
  down?: LayerWeightBuffer;
  gateUp?: LayerWeightBuffer;
  numExperts?: number;
  expertIntermediateSize?: number;
  gateUpBlocks?: GPUBuffer;
  gateUpScales?: GPUBuffer;
  gateUpBias?: GPUBuffer;
  downBlocks?: GPUBuffer;
  downScales?: GPUBuffer;
  downBias?: GPUBuffer;
}

/**
 * Router weights for MoE layers.
 */
export interface RouterWeights {
  weight: GPUBuffer | Float32Array | import('../../../gpu/weight-buffer.js').WeightBuffer;
  bias?: GPUBuffer | Float32Array | null;
  scale?: GPUBuffer | Float32Array | import('../../../gpu/weight-buffer.js').WeightBuffer | null;
  perExpertScale?: GPUBuffer | Float32Array | import('../../../gpu/weight-buffer.js').WeightBuffer | null;
}

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for weight buffer operations.
 */
export interface WeightBufferConfig {
  /** Whether RMSNorm uses (1 + weight) scaling at runtime */
  rmsNormWeightOffset: boolean;
}

/**
 * Debug flags for weight buffer operations.
 */
export interface WeightDebugFlags {
  normBufferTypeLogged?: boolean;
  normOffsetDebugDone?: boolean;
}

/**
 * Get layer weights from weights map with type narrowing.
 */
export function getLayerWeights(
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer>,
  key: string
): LayerWeights | null;

/**
 * Get or create GPU buffer for a weight tensor.
 */
export function getWeightBuffer(
  weight: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | ArrayBuffer,
  label: string,
  deviceOverride?: GPUDevice | null
): GPUBuffer | WeightBuffer;

/**
 * Get or create GPU buffer for RMSNorm weight tensor.
 */
export function getNormWeightBuffer(
  weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number } | CpuWeightBuffer,
  label: string,
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags,
  deviceOverride?: GPUDevice | null
): GPUBuffer;

export function getVectorTensor(
  weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number } | CpuWeightBuffer,
  label: string,
  length: number,
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags,
  deviceOverride?: GPUDevice | null
): { tensor: Tensor; owned: boolean };
