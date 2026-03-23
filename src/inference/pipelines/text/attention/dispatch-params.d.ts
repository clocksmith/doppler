/**
 * Attention Dispatch Parameters
 *
 * Shared dispatch parameter construction logic used by both run.js (immediate)
 * and record.js (batched) attention paths. Extracts KV cache state resolution,
 * sliding window normalization, kernel variant selection, attention scale
 * computation, and cached KV dtype/tensor creation.
 *
 * @module inference/pipelines/text/attention/dispatch-params
 */

import type { Tensor } from '../../../../gpu/tensor.js';
import type { AttentionState } from './types.js';

/**
 * Resolved KV cache state for attention dispatch.
 */
export interface KVCacheDispatchState {
  cachedK: GPUBuffer | undefined;
  cachedV: GPUBuffer | undefined;
  kvLenForAttention: number;
  causalForAttention: boolean;
  startPosForMask: number;
  kvStart: number;
  kvLayout: 'contiguous' | 'ring' | 'paged' | 'tiered' | 'bdpa';
  kvPageTable: GPUBuffer | null;
  kvPageSize: number;
  cachedKHot: GPUBuffer | undefined;
  cachedVHot: GPUBuffer | undefined;
  cachedKCold: GPUBuffer | undefined;
  cachedVCold: GPUBuffer | undefined;
  coldScalesK: GPUBuffer | null;
  coldScalesV: GPUBuffer | null;
  coldPackedStride: number;
  coldQuantMode: string;
  coldLen: number;
  hotLen: number;
  hotStart: number;
  hotWindow: number;
  coldPageTable: GPUBuffer | null;
  coldPageSize: number;
  bdpaBasisK: GPUBuffer | null;
  bdpaBasisV: GPUBuffer | null;
  bdpaPagedK: GPUBuffer | null;
  bdpaPagedV: GPUBuffer | null;
  bdpaIndex: GPUBuffer | null;
  bdpaBasisCount: number;
  hasCache: boolean;
  totalSeqLen: number;
}

/**
 * Config fields needed by buildAttentionDispatchParams.
 */
export interface DispatchParamsConfig {
  layerIdx: number;
  numTokens: number;
  isPrefill: boolean;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  hiddenSize: number;
  slidingWindow?: number | null;
  layerType?: string;
  layerTypes?: string[];
  queryPreAttnScalar?: number;
  causalAttention?: boolean;
  activationDtype?: string;
  kvCacheDtype?: string | null;
}

/**
 * Result from buildAttentionDispatchParams.
 */
export interface AttentionDispatchParams {
  effectiveSlidingWindow: number | null;
  attentionKernelVariant: string;
  attnScale: number;
  cachedKDtype: string;
  cachedVDtype: string;
  cachedKTensor: Tensor | null;
  cachedVTensor: Tensor | null;
  isTieredKernel: boolean;
  causalForAttention: boolean;
}

/**
 * Dtype info needed by buildAttentionInputsData.
 */
export interface DtypeInfo {
  useF16Activations: boolean;
  matmulOutputDtype: string;
}

/**
 * Check if a layer type string indicates sliding window attention.
 */
export function isSlidingLayerType(layerType: string | undefined): boolean;

/**
 * Resolve KV cache GPU buffers into a flat dispatch state object.
 *
 * Called after the KV cache has been updated (via updateFromGPU or
 * recordUpdateFromGPU) to extract buffer references, layout info,
 * and sequence lengths needed for attention dispatch.
 */
export function resolveKVCacheState(
  state: AttentionState,
  layerIdx: number,
  kTensor: Tensor,
  vTensor: Tensor,
  currentSeqLen: number,
  numTokens: number
): KVCacheDispatchState;

/**
 * Build attention dispatch parameters from the KV cache state and layer config.
 *
 * Applies tiered prefill fallback, sliding window normalization, kernel variant
 * selection, kvLen adjustments, attention scale computation, and cached KV
 * dtype/tensor creation. Mutates kvState in place for layout overrides.
 */
export function buildAttentionDispatchParams(
  config: DispatchParamsConfig,
  state: AttentionState,
  kTensor: Tensor,
  vTensor: Tensor,
  kvState: KVCacheDispatchState
): AttentionDispatchParams;

/**
 * Build the data object for recordAttentionInputs.
 *
 * Consolidates dtype, layout, and dimension info from config, KV state,
 * and dispatch params into the shape expected by recordAttentionInputs().
 */
export function buildAttentionInputsData(
  config: DispatchParamsConfig,
  input: Tensor,
  normed: Tensor,
  kvState: KVCacheDispatchState,
  dispatchParams: AttentionDispatchParams,
  dtypeInfo: DtypeInfo,
  usedFusedQKV: boolean,
  qTensor: Tensor | null,
  kTensor: Tensor | null,
  vTensor: Tensor | null
): Record<string, unknown>;
