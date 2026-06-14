

import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { SlidingWindowKVCache } from '../../../kv-cache.js';
import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';

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

function resolveDecoderEncoderWindow(encoderSeqLen, slidingWindow, layerType) {
  if (!isSlidingLayerType(layerType)) {
    return encoderSeqLen;
  }
  if (!Number.isFinite(slidingWindow) || slidingWindow <= 1) {
    return encoderSeqLen;
  }
  return Math.min(encoderSeqLen, Math.max(0, Math.trunc(slidingWindow) - 1));
}

function resolveDecoderKVCacheBuffers(state, layerIdx) {
  if (!state.kvCache?.hasGPUCache?.()) {
    throw new Error(
      `DiffusionGemma decoder attention at layer ${layerIdx} requires an initialized encoder KV cache.`
    );
  }
  const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
  const layout = gpuBuffers?.layout ?? state.kvCache?.layout ?? 'contiguous';
  if (layout !== 'contiguous' && layout !== undefined && layout !== null) {
    throw new Error(
      `DiffusionGemma decoder attention requires contiguous encoder KV cache at layer ${layerIdx}; got "${layout}".`
    );
  }
  if (!gpuBuffers?.keysGPU || !gpuBuffers?.valuesGPU) {
    throw new Error(
      `DiffusionGemma decoder attention missing contiguous GPU KV buffers at layer ${layerIdx}.`
    );
  }
  if (!Number.isFinite(gpuBuffers.seqLen) || gpuBuffers.seqLen < 0) {
    throw new Error(
      `DiffusionGemma decoder attention received invalid encoder KV length at layer ${layerIdx}: ${String(gpuBuffers.seqLen)}.`
    );
  }
  return gpuBuffers;
}

function copyKVRange(encoder, source, sourceOffset, target, targetOffset, size) {
  if (size <= 0) return;
  encoder.copyBufferToBuffer(source, sourceOffset, target, targetOffset, size);
}

export async function createDiffusionGemmaDecoderKVState({
  state,
  layerIdx,
  kTensor,
  vTensor,
  currentSeqLen,
  numTokens,
  numKVHeads,
  headDim,
  layerType,
  slidingWindow,
  kvDtype,
  recorder = null,
}) {
  const gpuBuffers = resolveDecoderKVCacheBuffers(state, layerIdx);
  const encoderSeqLen = Math.trunc(gpuBuffers.seqLen);
  if (encoderSeqLen !== currentSeqLen) {
    throw new Error(
      `DiffusionGemma decoder attention expected currentSeqLen=${encoderSeqLen} from encoder KV cache, ` +
      `got ${currentSeqLen}.`
    );
  }

  const encoderWindow = resolveDecoderEncoderWindow(encoderSeqLen, slidingWindow, layerType);
  const concatSeqLen = encoderWindow + numTokens;
  if (concatSeqLen <= 0) {
    throw new Error(`DiffusionGemma decoder attention has empty KV span at layer ${layerIdx}.`);
  }

  const dtype = kvDtype ?? kTensor.dtype;
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  const rowBytes = numKVHeads * headDim * bytesPerElement;
  const encoderStart = encoderSeqLen - encoderWindow;
  const encoderBytes = encoderWindow * rowBytes;
  const canvasBytes = numTokens * rowBytes;
  const concatBytes = concatSeqLen * rowBytes;
  let keysGPU = null;
  let valuesGPU = null;
  try {
    keysGPU = acquireBuffer(concatBytes, undefined, 'diffusion_gemma_decoder_keys');
    valuesGPU = acquireBuffer(concatBytes, undefined, 'diffusion_gemma_decoder_values');
    const device = recorder?.device ?? getDevice();
    if (!device) {
      throw new Error('DiffusionGemma decoder attention requires a GPU device.');
    }
    const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
    copyKVRange(
      encoder,
      gpuBuffers.keysGPU,
      encoderStart * rowBytes,
      keysGPU,
      0,
      encoderBytes
    );
    copyKVRange(
      encoder,
      gpuBuffers.valuesGPU,
      encoderStart * rowBytes,
      valuesGPU,
      0,
      encoderBytes
    );
    copyKVRange(
      encoder,
      kTensor.buffer,
      0,
      keysGPU,
      encoderBytes,
      canvasBytes
    );
    copyKVRange(
      encoder,
      vTensor.buffer,
      0,
      valuesGPU,
      encoderBytes,
      canvasBytes
    );
    if (!recorder) {
      device.queue.submit([encoder.finish()]);
    }
  } catch (error) {
    if (keysGPU) releaseBuffer(keysGPU);
    if (valuesGPU && valuesGPU !== keysGPU) releaseBuffer(valuesGPU);
    throw error;
  }

  return {
    cachedK: keysGPU,
    cachedV: valuesGPU,
    kvLenForAttention: concatSeqLen,
    causalForAttention: false,
    startPosForMask: encoderSeqLen,
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
    hasCache: true,
    totalSeqLen: concatSeqLen,
    diffusionGemmaDecoder: true,
    ownedBuffers: [keysGPU, valuesGPU],
    encoderSeqLen,
    encoderWindow,
  };
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
