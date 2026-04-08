

import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { SlidingWindowKVCache } from '../../../kv-cache.js';

// ============================================================================
// Layer Type Helpers
// ============================================================================

function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

export function isSlidingLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding';
}

// ============================================================================
// KV Cache State Resolution
// ============================================================================

export function resolveKVCacheState(state, layerIdx, kTensor, vTensor, currentSeqLen, numTokens) {
  const kvState = {
    cachedK: undefined,
    cachedV: undefined,
    kvLenForAttention: currentSeqLen + numTokens,
    causalForAttention: true,
    startPosForMask: currentSeqLen,
    kvStart: 0,
    kvLayout: 'contiguous',
    kvPageTable: null,
    kvPageSize: 0,
    cachedKHot: undefined,
    cachedVHot: undefined,
    cachedKCold: undefined,
    cachedVCold: undefined,
    coldScalesK: null,
    coldScalesV: null,
    coldPackedStride: 0,
    coldQuantMode: 'none',
    coldLen: 0,
    hotLen: 0,
    hotStart: 0,
    hotWindow: 0,
    coldPageTable: null,
    coldPageSize: 0,
    bdpaBasisK: null,
    bdpaBasisV: null,
    bdpaPagedK: null,
    bdpaPagedV: null,
    bdpaIndex: null,
    bdpaBasisCount: 0,
    hasCache: false,
    totalSeqLen: currentSeqLen + numTokens,
  };

  kvState.hasCache = !!state.kvCache?.hasGPUCache?.();

  if (!kvState.hasCache) {
    kvState.cachedK = kTensor.buffer;
    kvState.cachedV = vTensor.buffer;
    kvState.kvLenForAttention = numTokens;
    kvState.startPosForMask = 0;
    return kvState;
  }

  const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
  if (gpuBuffers?.layout === 'tiered') {
    kvState.cachedKHot = gpuBuffers.hotKeysGPU;
    kvState.cachedVHot = gpuBuffers.hotValuesGPU;
    kvState.cachedKCold = gpuBuffers.coldKeysGPU;
    kvState.cachedVCold = gpuBuffers.coldValuesGPU;
    kvState.coldScalesK = gpuBuffers.coldScalesKGPU ?? null;
    kvState.coldScalesV = gpuBuffers.coldScalesVGPU ?? null;
    kvState.coldPackedStride = gpuBuffers.coldPackedStride ?? 0;
    kvState.coldQuantMode = gpuBuffers.coldQuantMode ?? 'none';
    kvState.hotLen = gpuBuffers.hotSeqLen ?? 0;
    kvState.coldLen = gpuBuffers.coldSeqLen ?? 0;
    kvState.hotStart = gpuBuffers.hotStart ?? 0;
    kvState.hotWindow = gpuBuffers.hotWindow ?? 0;
    kvState.coldPageTable = gpuBuffers.coldPageTableGPU ?? null;
    kvState.coldPageSize = gpuBuffers.coldPageSize ?? state.kvCache.coldPageSize ?? 0;
    kvState.kvLenForAttention = kvState.coldLen + kvState.hotLen;
    kvState.kvLayout = 'tiered';
    // TurboQuant shared buffers
    kvState.rotationMatrixBuffer = gpuBuffers.rotationMatrixBuffer ?? null;
    kvState.codebookCentroidsBuffer = gpuBuffers.codebookCentroidsBuffer ?? null;
    // TurboQuant prod buffers
    kvState.residualKGPU = gpuBuffers.residualKGPU ?? null;
    kvState.residualVGPU = gpuBuffers.residualVGPU ?? null;
    kvState.residualNormsKGPU = gpuBuffers.residualNormsKGPU ?? null;
    kvState.residualNormsVGPU = gpuBuffers.residualNormsVGPU ?? null;
    kvState.qjlMatrixBuffer = gpuBuffers.qjlMatrixBuffer ?? null;
    kvState.residualPackedStride = gpuBuffers.residualPackedStride ?? 0;
  } else if (gpuBuffers?.layout === 'contiguous_quantized') {
    kvState.kvLayout = 'contiguous_quantized';
    kvState.kvLenForAttention = gpuBuffers.seqLen;
    kvState.cachedKCold = gpuBuffers.keysPackedGPU;
    kvState.cachedVCold = gpuBuffers.valuesPackedGPU;
    kvState.coldScalesK = gpuBuffers.scalesKGPU ?? null;
    kvState.coldScalesV = gpuBuffers.scalesVGPU ?? null;
    kvState.coldPackedStride = gpuBuffers.packedStride ?? 0;
    kvState.coldQuantMode = gpuBuffers.quantMode ?? 'turboquant';
    kvState.rotationMatrixBuffer = gpuBuffers.rotationMatrixBuffer ?? null;
    kvState.codebookCentroidsBuffer = gpuBuffers.codebookCentroidsBuffer ?? null;
    // Prod-mode buffers
    kvState.residualKGPU = gpuBuffers.residualKGPU ?? null;
    kvState.residualVGPU = gpuBuffers.residualVGPU ?? null;
    kvState.residualNormsKGPU = gpuBuffers.residualNormsKGPU ?? null;
    kvState.residualNormsVGPU = gpuBuffers.residualNormsVGPU ?? null;
    kvState.qjlMatrixBuffer = gpuBuffers.qjlMatrixBuffer ?? null;
    kvState.residualPackedStride = gpuBuffers.residualPackedStride ?? 0;
    kvState.prodMode = gpuBuffers.prodMode === true;
  } else if (gpuBuffers?.layout === 'bdpa') {
    kvState.kvLayout = 'bdpa';
    kvState.kvLenForAttention = gpuBuffers.seqLen;
    kvState.bdpaBasisK = gpuBuffers.basisGPU.k;
    kvState.bdpaBasisV = gpuBuffers.basisGPU.v;
    kvState.bdpaPagedK = gpuBuffers.pagedGPU.k;
    kvState.bdpaPagedV = gpuBuffers.pagedGPU.v;
    kvState.bdpaIndex = gpuBuffers.indexGPU;
    kvState.bdpaBasisCount = gpuBuffers.numBasisVectors ?? state.kvCache.basisVocabSize;
  } else {
    kvState.cachedK = gpuBuffers.keysGPU;
    kvState.cachedV = gpuBuffers.valuesGPU;
    kvState.kvLenForAttention = gpuBuffers.seqLen;
    kvState.kvPageTable = gpuBuffers.pageTableGPU ?? null;
    kvState.kvPageSize = gpuBuffers.pageSize ?? state.kvCache.pageSize ?? 0;
    if (gpuBuffers?.layout === 'ring' || state.kvCache instanceof SlidingWindowKVCache) {
      kvState.kvLayout = 'ring';
    } else if (state.kvCache.layout === 'paged') {
      kvState.kvLayout = 'paged';
    }
  }

  return kvState;
}

// ============================================================================
// Dispatch Parameter Construction
// ============================================================================

export function buildAttentionDispatchParams(config, state, kTensor, vTensor, kvState) {
  const {
    numTokens, slidingWindow, layerType, headDim, queryPreAttnScalar, numKVHeads,
  } = config;
  const resolvedKvCacheDtype = config.kvCacheDtype ?? state.kvCache?.kvDtype ?? null;

  // Tiered prefill fallback: tiered layout does not support prefill (numTokens > 1)
  let prefillFallbackNeedsCast = false;
  if (kvState.kvLayout === 'tiered' && numTokens > 1) {
    kvState.kvLayout = 'contiguous';
    kvState.kvLenForAttention = numTokens;
    kvState.startPosForMask = 0;
    kvState.cachedKHot = null;
    kvState.cachedVHot = null;
    kvState.cachedKCold = null;
    kvState.cachedVCold = null;
    kvState.coldQuantMode = 'none';
    prefillFallbackNeedsCast = true;
  }

  // Contiguous quantized prefill fallback: decode-only kernel, use raw K/V for prefill
  if (kvState.kvLayout === 'contiguous_quantized' && numTokens > 1) {
    kvState.kvLayout = 'contiguous';
    kvState.kvLenForAttention = numTokens;
    kvState.startPosForMask = 0;
    kvState.cachedKCold = null;
    kvState.cachedVCold = null;
    kvState.coldQuantMode = 'none';
    prefillFallbackNeedsCast = true;
  }

  // Sliding window
  const hasSlidingWindow = Number.isFinite(slidingWindow) && slidingWindow > 0;
  const hasLayerTypes = Array.isArray(config.layerTypes);
  const isLayerSliding = isSlidingLayerType(layerType) || (!hasLayerTypes && hasSlidingWindow);
  const effectiveSlidingWindow = isLayerSliding ? slidingWindow : null;
  const canWindow = kvState.hasCache && effectiveSlidingWindow;

  // Kernel variant selection
  const attentionKernelVariant = selectRuleValue('inference', 'attention', 'attentionKernelVariant', {
    kvLayout: kvState.kvLayout,
    numTokens,
    coldQuantMode: kvState.coldQuantMode,
  });

  // Variant-driven overrides
  if (attentionKernelVariant === 'contiguous' && kvState.kvLayout === 'tiered') {
    kvState.kvLayout = 'contiguous';
    kvState.cachedK = kTensor.buffer;
    kvState.cachedV = vTensor.buffer;
    kvState.kvLenForAttention = numTokens;
    kvState.startPosForMask = 0;
    kvState.cachedKHot = null;
    kvState.cachedVHot = null;
    kvState.cachedKCold = null;
    kvState.cachedVCold = null;
    kvState.coldQuantMode = 'none';
  }

  if (attentionKernelVariant !== 'tiered' && attentionKernelVariant !== 'tieredQuant') {
    if (canWindow && kvState.kvLenForAttention > effectiveSlidingWindow) {
      kvState.kvLenForAttention = effectiveSlidingWindow;
    }
    if (kvState.hasCache && (kvState.kvLayout === 'ring' || (canWindow && kvState.kvLenForAttention < kvState.totalSeqLen))) {
      kvState.kvStart = Math.max(0, kvState.totalSeqLen - kvState.kvLenForAttention);
    }
  }

  if (kvState.kvLenForAttention <= 0) {
    throw new Error(`Invalid kvLen ${kvState.kvLenForAttention} at layer ${config.layerIdx}`);
  }

  // Attention scale
  const attnScale = queryPreAttnScalar ? 1.0 / Math.sqrt(queryPreAttnScalar) : 1.0 / Math.sqrt(headDim);

  // Cached K/V dtypes
  const cachedKDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: resolvedKvCacheDtype,
    fallback: kTensor.dtype,
  });
  const cachedVDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: resolvedKvCacheDtype,
    fallback: vTensor.dtype,
  });

  // Cached K/V tensors (null for tiered, contiguousQuant, and prefill-fallback paths)
  const isTieredKernel = attentionKernelVariant === 'tiered' || attentionKernelVariant === 'tieredQuant';
  const isContiguousQuantKernel = attentionKernelVariant === 'contiguousQuant';
  const skipCachedKVTensors = isTieredKernel || isContiguousQuantKernel || prefillFallbackNeedsCast;
  const cachedKTensor = skipCachedKVTensors
    ? null
    : createTensor(kvState.cachedK, cachedKDtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = skipCachedKVTensors
    ? null
    : createTensor(kvState.cachedV, cachedVDtype, [kvState.kvLenForAttention, numKVHeads * headDim], 'cached_V');

  return {
    effectiveSlidingWindow,
    attentionKernelVariant,
    attnScale,
    cachedKDtype,
    cachedVDtype,
    cachedKTensor,
    cachedVTensor,
    isTieredKernel,
    prefillFallbackNeedsCast,
    causalForAttention: config.causalAttention !== false,
  };
}

// ============================================================================
// recordAttentionInputs Data Builder
// ============================================================================

export function buildAttentionInputsData(config, input, normed, kvState, dispatchParams, dtypeInfo, usedFusedQKV, qTensor, kTensor, vTensor) {
  const { isPrefill, layerIdx, numTokens, numHeads, numKVHeads, headDim } = config;
  const { useF16Activations, matmulOutputDtype } = dtypeInfo;
  const { cachedKDtype, cachedVDtype } = dispatchParams;
  return {
    phase: isPrefill ? 'prefill' : 'decode',
    layerIdx,
    numTokens,
    kvLen: kvState.kvLenForAttention,
    numHeads,
    numKVHeads,
    headDim,
    activationDtype: config.activationDtype ?? null,
    inputDtype: input.dtype,
    normedDtype: normed.dtype,
    useF16Activations,
    matmulOutputDtype,
    kvCacheDtype: config.kvCacheDtype ?? null,
    cachedKDtype,
    cachedVDtype,
    qDtype: qTensor?.dtype ?? null,
    kDtype: kTensor?.dtype ?? null,
    vDtype: vTensor?.dtype ?? null,
    useFusedQKV: usedFusedQKV,
    kvStart: kvState.kvStart,
    kvLayout: kvState.kvLayout,
    kvPageSize: kvState.kvLayout === 'tiered' ? (kvState.coldPageSize || null) : (kvState.kvPageSize || null),
    hotLen: kvState.kvLayout === 'tiered' ? kvState.hotLen : null,
    coldLen: kvState.kvLayout === 'tiered' ? kvState.coldLen : null,
    hotWindow: kvState.kvLayout === 'tiered' ? kvState.hotWindow : null,
    hotStart: kvState.kvLayout === 'tiered' ? kvState.hotStart : null,
    coldQuantMode: kvState.kvLayout === 'tiered' ? kvState.coldQuantMode : null,
  };
}
