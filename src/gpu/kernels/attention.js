

import { getDevice, getDeviceEpoch, getDeviceLimits, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { TILE_SIZES } from './constants.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { createUniformBufferWithView, getKernelConfig, hasRequiredFeatures } from './utils.js';
import { dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { log, trace } from '../../debug/index.js';
import { getKernelPathAttentionVariant, getKernelPathStrict } from '../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../kernel-selection-log.js';

// Track if we've logged the attention tier selection (avoid spam)
let loggedAttentionTier = false;

function getRequiredVariantMaxKVLen(operation, variant, errorLabel) {
  const config = getKernelConfig(operation, variant);
  const maxKVLen = config.variantMetadata?.maxKVLen;
  if (!Number.isFinite(maxKVLen)) {
    throw new Error(`Kernel config missing ${errorLabel} maxKVLen`);
  }
  return maxKVLen;
}

function getChunkedMaxKVLen() {
  return getRequiredVariantMaxKVLen('attention', 'decode_chunked_f16kv', 'attention.decode_chunked_f16kv');
}

function getTieredMaxKVLen() {
  return getRequiredVariantMaxKVLen('attention_tiered', 'decode_tiered_f16', 'attention_tiered.decode_tiered_f16');
}

function getTieredQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_tiered_quant',
    'decode_tiered_int8_f16kv',
    'attention_tiered_quant.decode_tiered_int8_f16kv'
  );
}

function getContiguousQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_contiguous_quant',
    'decode_contiguous_turboquant_f16kv',
    'attention_contiguous_quant.decode_contiguous_turboquant_f16kv'
  );
}


let kvLenFallbackBuffer = null;
let kvLenFallbackBufferEpoch = -1;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;


function getKvLenFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!kvLenFallbackBuffer || kvLenFallbackBufferEpoch !== epoch) {
    kvLenFallbackBuffer = device.createBuffer({
      label: 'attention_kv_len_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(kvLenFallbackBuffer, 0, new Uint32Array([0]));
    kvLenFallbackBufferEpoch = epoch;
  }
  return kvLenFallbackBuffer;
}

let pageTableFallbackBuffer = null;
let pageTableFallbackBufferEpoch = -1;

function getPageTableFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!pageTableFallbackBuffer || pageTableFallbackBufferEpoch !== epoch) {
    pageTableFallbackBuffer = device.createBuffer({
      label: 'attention_page_table_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pageTableFallbackBuffer, 0, new Uint32Array([0]));
    pageTableFallbackBufferEpoch = epoch;
  }
  return pageTableFallbackBuffer;
}




class AttentionKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention', variant);
  }


  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }


  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
}

class AttentionTieredKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered', variant);
  }


  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered');
  }


  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered');
  }
}

class AttentionTieredQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered_quant', variant);
  }


  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }


  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }
}

class AttentionContiguousQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_contiguous_quant', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }
}

class AttentionBDPAKernel extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor('attention_bdpa', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_bdpa');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_bdpa');
  }
}


function selectAttentionTier(
  headDim,
  seqLen,
  useF16KV,
  forcedTier,
  sharedLimit,
  caps
) {
  const isDecode = seqLen === 1;
  const thresholds = getKernelThresholds().attention;
  const largeRequired = useF16KV
    ? thresholds.largeSharedF16
    : thresholds.largeSharedF32;
  const canLarge =
    headDim <= thresholds.largeMaxHeadDim &&
    sharedLimit >= largeRequired;
  const smallRequired = useF16KV
    ? thresholds.smallSharedF16
    : thresholds.smallSharedF32;
  const canSmall =
    headDim <= thresholds.smallMaxHeadDim &&
    sharedLimit >= smallRequired;
  const canSubgroup =
    caps.hasSubgroups &&
    headDim <= thresholds.subgroupMaxHeadDim &&
    sharedLimit >= thresholds.subgroupShared &&
    isDecode;


  let tier = forcedTier;
  let reason = forcedTier ? `forced:${forcedTier}` : '';

  if (tier === 'tiled_large' && !canLarge) {
    throw new Error(`Requested tiled_large but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
  }
  if (tier === 'tiled_small' && !canSmall) {
    throw new Error(`Requested tiled_small but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
  }
  if (tier === 'subgroup' && !canSubgroup) {
    throw new Error(`Requested subgroup attention but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}, subgroups=${caps.hasSubgroups}).`);
  }

  if (!tier) {
    tier = selectKernelRuleValue('attention', 'tier', { canSubgroup, canLarge, canSmall, isDecode });
    if (!reason) {
      if (canSubgroup) {
        reason = 'subgroup_capable';
      } else if (canLarge) {
        reason = 'tiled_large_capable';
      } else if (canSmall) {
        reason = 'tiled_small_capable';
      } else if (isDecode) {
        reason = 'decode_streaming_fallback';
      } else {
        reason = 'streaming_fallback';
      }
    }
    if (tier === 'subgroup' && !loggedAttentionTier) {
      trace.attn(0, `Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
      loggedAttentionTier = true;
    }
  }

  return { tier, reason };
}

// Track if we've logged chunked kernel selection
let loggedChunkedKernel = false;


function resolveAttentionVariant(
  tier,
  isDecode,
  useF16KV,
  useF16Q,
  numHeads,
  headDim,
  kvLen,
  isPaged,
  caps,
  sharedLimit
) {
  const base = selectKernelRuleValue('attention', 'phase', { isDecode });
  const useF16 = useF16KV && useF16Q;
  const suffix = selectKernelRuleValue('attention', 'suffix', { useF16, useF16KV });

  // Check if chunked kernel is viable:
  // - Decode only (seqLen=1)
  // - F16 KV cache
  // - Large headDim (parallelizes across dimensions)
  // - KV length within shared memory limit (from kernel config)
  const chunkedMaxKVLen = getChunkedMaxKVLen();
  const minHeadDimForChunked = getKernelThresholds().attention.minHeadDimForChunked;
  const canUseChunked = isDecode && useF16KV && headDim >= minHeadDimForChunked && kvLen <= chunkedMaxKVLen;
  const decodeSubgroupMaxKVLen = chunkedMaxKVLen;
  const decodeSubgroupMaxHeadDim = getKernelThresholds().attention.subgroupMaxHeadDim;
  const canUseDecodeSubgroup = isDecode && !useF16KV && !useF16Q && headDim <= decodeSubgroupMaxHeadDim && kvLen <= decodeSubgroupMaxKVLen;
  const canUseDecodeOptimized = isDecode
    && useF16KV
    && caps.hasF16
    && caps.hasSubgroups
    && headDim <= decodeSubgroupMaxHeadDim
    && sharedLimit >= getKernelThresholds().attention.subgroupShared;
  const chunkedVariant = selectKernelRuleValue('attention', 'chunkedVariant', { useF16 });
  const pagedVariant = selectKernelRuleValue('attention', 'pagedVariant', { useF16 });
  const optimizedVariant = selectKernelRuleValue('attention', 'optimizedVariant', { useF16 });
  const variant = selectKernelRuleValue(
    'attention',
    'variant',
    {
      tier,
      useF16KV,
      canUseChunked,
      canUseDecodeSubgroup,
      canUseDecodeOptimized,
      base,
      suffix,
      chunkedVariant,
      pagedVariant,
      optimizedVariant,
      isPaged,
      isDecode,
    }
  );

  if (variant === chunkedVariant && !loggedChunkedKernel) {
    trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=${!useF16Q})`);
    loggedChunkedKernel = true;
  }

  return variant;
}


function calculateAttentionWorkgroups(tier, seqLen, numHeads) {
  if (tier === 'subgroup') {
    return numHeads;
  }
  if (tier === 'streaming') {
    return seqLen * numHeads;
  }
  if (tier === 'tiled_large') {
    return Math.ceil(seqLen / TILE_SIZES.ATTENTION_LARGE_BLOCK_SIZE) * numHeads;
  }
  return Math.ceil(seqLen / TILE_SIZES.ATTENTION_SMALL_BLOCK_SIZE) * numHeads;
}


function inferAttentionTierFromVariant(variant) {
  const config = getKernelConfig('attention', variant);
  const tier = config.variantMetadata?.tier;
  if (!tier) {
    throw new Error(`Attention kernel "${variant}" missing variantMetadata.tier in registry.`);
  }
  return tier;
}


function validateAttentionVariant(
  variant,
  isDecode,
  useF16KV,
  useF16Q,
  caps,
  headDim,
  kvLen,
  sharedLimit
) {
  const normalized = variant.trim();

  let config;
  try {
    config = getKernelConfig('attention', normalized);
  } catch {
    throw new Error(`Unknown attention kernel variant "${variant}".`);
  }

  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const expectsF16KV = normalized.includes('_f16kv');
  const expectsF16 = normalized.includes('_f16') && !expectsF16KV;
  if (expectsF16) {
    if (!(useF16KV && useF16Q)) {
      const kvLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16KV });
      const qLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Q });
      throw new Error(`Attention kernel "${variant}" requires f16 Q/K/V but got Q=${qLabel}, KV=${kvLabel}.`);
    }
  } else if (expectsF16KV) {
    if (!useF16KV || useF16Q) {
      const kvLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16KV });
      const qLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Q });
      throw new Error(`Attention kernel "${variant}" requires f32 Q with f16 KV but got Q=${qLabel}, KV=${kvLabel}.`);
    }
  } else {
    if (useF16KV || useF16Q) {
      const kvLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16KV });
      const qLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Q });
      throw new Error(`Attention kernel "${variant}" requires f32 Q/K/V but got Q=${qLabel}, KV=${kvLabel}.`);
    }
  }

  const isDecodeVariant = normalized.startsWith('decode');
  const isPrefillVariant = normalized.startsWith('prefill');
  if (isDecode && isPrefillVariant) {
    throw new Error(`Attention kernel "${variant}" is prefill-only but decode requested.`);
  }
  if (!isDecode && isDecodeVariant) {
    throw new Error(`Attention kernel "${variant}" is decode-only but prefill requested.`);
  }

  const thresholds = getKernelThresholds().attention;
  const chunkedMaxKVLen = getChunkedMaxKVLen();
  const isChunked = normalized.startsWith('decode_chunked');
  if (isChunked) {
    const minHeadDimForChunked = thresholds.minHeadDimForChunked;
    if (headDim < minHeadDimForChunked) {
      throw new Error(`Attention kernel "${variant}" requires headDim >= ${minHeadDimForChunked} but got ${headDim}.`);
    }
    if (kvLen > chunkedMaxKVLen) {
      throw new Error(`Attention kernel "${variant}" requires kvLen <= ${chunkedMaxKVLen} but got ${kvLen}.`);
    }
  }

  if (normalized === 'decode_subgroup') {
    if (!caps.hasSubgroups) {
      throw new Error(`Attention kernel "${variant}" requires subgroup support.`);
    }
    if (headDim > thresholds.subgroupMaxHeadDim) {
      throw new Error(`Attention kernel "${variant}" requires headDim <= ${thresholds.subgroupMaxHeadDim} but got ${headDim}.`);
    }
    if (kvLen > chunkedMaxKVLen) {
      throw new Error(`Attention kernel "${variant}" requires kvLen <= ${chunkedMaxKVLen} but got ${kvLen}.`);
    }
    if (sharedLimit < thresholds.subgroupShared) {
      throw new Error(`Attention kernel "${variant}" requires shared >= ${thresholds.subgroupShared} but got ${sharedLimit}.`);
    }
  }

  if (normalized.startsWith('decode_online')) {
    const maxHeadDim = config.variantMetadata?.maxHeadDim ?? thresholds.subgroupMaxHeadDim;
    if (!caps.hasSubgroups) {
      throw new Error(`Attention kernel "${variant}" requires subgroup support.`);
    }
    if (headDim > maxHeadDim) {
      throw new Error(`Attention kernel "${variant}" requires headDim <= ${maxHeadDim} but got ${headDim}.`);
    }
    if (sharedLimit < thresholds.subgroupShared) {
      throw new Error(`Attention kernel "${variant}" requires shared >= ${thresholds.subgroupShared} but got ${sharedLimit}.`);
    }
  }

  if (normalized.startsWith('prefill') || normalized.startsWith('decode')) {
    const isSmall = normalized.includes('_small');
    const isStreaming = normalized.includes('_streaming');
    const isTiled = !isStreaming
      && !normalized.startsWith('decode_subgroup')
      && !normalized.startsWith('decode_online')
      && !isChunked;
    if (isTiled) {
      const metadata = config.variantMetadata ?? {};
      const requiredShared = metadata.requiredShared ?? (
        isSmall
          ? (useF16KV ? thresholds.smallSharedF16 : thresholds.smallSharedF32)
          : (useF16KV ? thresholds.largeSharedF16 : thresholds.largeSharedF32)
      );
      const maxHeadDim = metadata.maxHeadDim ?? (isSmall ? thresholds.smallMaxHeadDim : thresholds.largeMaxHeadDim);
      const minHeadDim = metadata.minHeadDim ?? 0;
      const exactHeadDim = metadata.exactHeadDim;
      if (Number.isFinite(exactHeadDim) && headDim !== exactHeadDim) {
        throw new Error(`Attention kernel "${variant}" requires headDim == ${exactHeadDim} but got ${headDim}.`);
      }
      if (headDim < minHeadDim) {
        throw new Error(`Attention kernel "${variant}" requires headDim >= ${minHeadDim} but got ${headDim}.`);
      }
      if (headDim > maxHeadDim) {
        throw new Error(`Attention kernel "${variant}" requires headDim <= ${maxHeadDim} but got ${headDim}.`);
      }
      if (sharedLimit < requiredShared) {
        throw new Error(`Attention kernel "${variant}" requires shared >= ${requiredShared} but got ${sharedLimit}.`);
      }
    }
  }

  return normalized;
}


function resolveAttentionPlan(
  seqLen,
  kvLen,
  headDim,
  numHeads,
  kvDtype,
  qDtype,
  sharedLimit,
  caps,
  layerIdx,
  isPaged,
  kernelPath
) {
  const useF16KV = kvDtype === 'f16';
  const useF16Q = qDtype === 'f16';
  const isDecode = seqLen === 1;
  const phase = selectKernelRuleValue('attention', 'phase', { isDecode });
  const pathVariant = getKernelPathAttentionVariant(phase, layerIdx, kernelPath);
  const strictPath = getKernelPathStrict();

  if (pathVariant) {
    let variantOverride;
    try {
      variantOverride = validateAttentionVariant(
        pathVariant,
        isDecode,
        useF16KV,
        useF16Q,
        caps,
        headDim,
        kvLen,
        sharedLimit
      );
    } catch (error) {
      if (strictPath) {
        throw error;
      }
      const reason = error instanceof Error ? error.message : String(error);
      log.warn(
        'Attention',
        `Kernel path override "${pathVariant}" rejected; falling back to capability selection: ${reason}`
      );
      const adaptiveSelection = selectAttentionTier(headDim, seqLen, useF16KV, null, sharedLimit, caps);
      const adaptiveVariant = resolveAttentionVariant(
        adaptiveSelection.tier,
        isDecode,
        useF16KV,
        useF16Q,
        numHeads,
        headDim,
        kvLen,
        isPaged,
        caps,
        sharedLimit
      );
      const workgroups = calculateAttentionWorkgroups(adaptiveSelection.tier, seqLen, numHeads);
      logKernelSelectionOnce('attention', {
        variant: adaptiveVariant,
        reason: `path_override_fallback:${adaptiveSelection.tier}`,
      });
      return {
        tier: adaptiveSelection.tier,
        variant: adaptiveVariant,
        workgroups,
        useF16KV,
        isDecode,
      };
    }
    let selectionReason = 'path_override';

    if (!isDecode && variantOverride.startsWith('prefill_streaming') && seqLen <= 64) {
      const adaptivePrefillVariant = variantOverride.endsWith('_f16kv')
        ? 'prefill_f16kv'
        : variantOverride.endsWith('_f16')
          ? 'prefill_f16'
          : 'prefill';
      try {
        const validatedAdaptive = validateAttentionVariant(
          adaptivePrefillVariant,
          isDecode,
          useF16KV,
          useF16Q,
          caps,
          headDim,
          kvLen,
          sharedLimit
        );
        if (validatedAdaptive !== variantOverride) {
          variantOverride = validatedAdaptive;
          selectionReason = 'path_override_adaptive_prefill';
        }
      } catch {
        // Keep original strict-path variant when adaptive fallback is not valid.
      }
    }

    const tier = inferAttentionTierFromVariant(variantOverride);
    const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);
    logKernelSelectionOnce('attention', {
      variant: variantOverride,
      reason: `${selectionReason}:${tier}`,
    });
    return { tier, variant: variantOverride, workgroups, useF16KV, isDecode };
  }

  const selection = selectAttentionTier(headDim, seqLen, useF16KV, null, sharedLimit, caps);
  const tier = selection.tier;
  const variant = resolveAttentionVariant(
    tier,
    isDecode,
    useF16KV,
    useF16Q,
    numHeads,
    headDim,
    kvLen,
    isPaged,
    caps,
    sharedLimit
  );
  const validatedVariant = validateAttentionVariant(
    variant,
    isDecode,
    useF16KV,
    useF16Q,
    caps,
    headDim,
    kvLen,
    sharedLimit
  );
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);

  logKernelSelectionOnce('attention', {
    variant: validatedVariant,
    reason: selection.reason,
  });

  return { tier, variant: validatedVariant, workgroups, useF16KV, isDecode };
}

export function resolveAttentionPlanForTest(
  seqLen,
  kvLen,
  headDim,
  numHeads,
  kvDtype,
  qDtype,
  sharedLimit,
  caps,
  layerIdx,
  isPaged = false,
  kernelPath = null
) {
  return resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    qDtype,
    sharedLimit,
    caps,
    layerIdx,
    isPaged,
    kernelPath
  );
}


function createAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_uniforms',
    80,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true); // Gemma 2: 50.0, 0 = disabled
      view.setUint32(36, params.slidingWindow, true); // Sliding window size, 0 = disabled
      view.setUint32(40, params.kvLenSource, true); // 0 = uniform kvLen, 1 = buffer
      view.setUint32(44, params.kvStart ?? 0, true);
      view.setUint32(48, params.pageSize ?? 0, true);
      view.setUint32(52, params.kvLayout ?? 0, true);
      view.setUint32(56, params.bidirectionalSpanStart ?? 0, true);
      view.setUint32(60, params.bidirectionalSpanLength ?? 0, true);
      view.setUint32(64, 0, true);
      view.setUint32(68, 0, true);
      view.setUint32(72, 0, true);
    },
    recorder,
    device
  );
}

function createTieredAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_uniforms',
    80,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.coldLen, true);
      view.setUint32(16, params.hotLen, true);
      view.setUint32(20, params.seqLen, true);
      view.setFloat32(24, params.scale, true);
      view.setUint32(28, params.causal ? 1 : 0, true);
      view.setUint32(32, params.startPos, true);
      view.setFloat32(36, params.attnSoftcap, true);
      view.setUint32(40, params.slidingWindow, true);
      view.setUint32(44, params.hotWindow, true);
      view.setUint32(48, params.hotStart, true);
      view.setUint32(52, params.coldPageSize, true);
      view.setUint32(56, params.coldLayout ?? 0, true);
      view.setUint32(60, params.hotLayout ?? 1, true);
      view.setUint32(64, 0, true);
    },
    recorder,
    device
  );
}

function createTieredQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_quant_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.coldLen, true);
      view.setUint32(16, params.hotLen, true);
      view.setUint32(20, params.seqLen, true);
      view.setFloat32(24, params.scale, true);
      view.setUint32(28, params.causal ? 1 : 0, true);
      view.setUint32(32, params.startPos, true);
      view.setFloat32(36, params.attnSoftcap, true);
      view.setUint32(40, params.slidingWindow, true);
      view.setUint32(44, params.hotWindow, true);
      view.setUint32(48, params.hotStart, true);
      view.setUint32(52, params.packedStride, true);
      view.setUint32(56, 0, true);
    },
    recorder,
    device
  );
}

function createContiguousQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  const hasProdFields = params.packedStrideMSE != null;
  const size = hasProdFields ? 64 : 48;
  return createUniformBufferWithView(
    'attention_contiguous_quant_uniforms',
    size,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      if (hasProdFields) {
        view.setUint32(40, params.packedStrideMSE, true);
        view.setUint32(44, params.packedStrideResidual, true);
        view.setUint32(48, 0, true);
        view.setUint32(52, 0, true);
        view.setUint32(56, 0, true);
        view.setUint32(60, 0, true);
      } else {
        view.setUint32(40, params.packedStride, true);
        view.setUint32(44, 0, true);
      }
    },
    recorder,
    device
  );
}

function createBDPAAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_bdpa_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      view.setUint32(40, 0, true); // padding
      view.setUint32(44, 0, true); // padding
      view.setUint32(48, 0, true); // padding
      view.setUint32(52, 0, true); // padding
      view.setUint32(56, 0, true); // padding
      view.setUint32(60, 0, true); // padding
    },
    recorder,
    device
  );
}

function resolveAttentionExecution(recorder) {
  return {
    recorder: recorder || null,
    device: recorder?.device || getDevice(),
  };
}

function assertAttentionBindGroupBuffer(kernelName, variant, bindingIndex, bindingLabel, buffer, details = []) {
  const isGpuBuffer = buffer && (
    typeof GPUBuffer === 'undefined'
      ? true
      : buffer instanceof GPUBuffer
  );
  if (isGpuBuffer) {
    return;
  }
  const detailText = details.filter(Boolean).join(', ');
  throw new Error(
    `[${kernelName}] variant="${variant}" binding ${bindingIndex} "${bindingLabel}" requires a GPUBuffer` +
    (detailText ? ` (${detailText})` : '') +
    '.'
  );
}

function releaseAttentionUniform(execution, uniformBuffer) {
  if (!execution.recorder) {
    releaseUniformBuffer(uniformBuffer);
  }
}

function dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, workgroups) {
  if (execution.recorder) {
    kernel.record(execution.recorder, pipeline, bindGroup, workgroups);
    return;
  }
  kernel.dispatch(pipeline, bindGroup, workgroups);
}

async function executeAttentionBDPA(
  recorder,
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    ropeCos = null,
    ropeSin = null,
  } = options;

  if (seqLen !== 1) {
    throw new Error(`BDPA attention currently supports decode only (seqLen=1), got seqLen=${seqLen}.`);
  }
  if (Q.dtype !== 'f16' || basisK.dtype !== 'f16' || basisV.dtype !== 'f16') {
    throw new Error(`BDPA attention requires f16 Q/basis tensors; got Q=${Q.dtype}, basisK=${basisK.dtype}, basisV=${basisV.dtype}.`);
  }
  if (!(ropeCos instanceof GPUBuffer) || !(ropeSin instanceof GPUBuffer)) {
    throw new Error('BDPA attention requires GPU ropeCos/ropeSin buffers.');
  }

  const variant = 'decode_bdpa_f16';
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_bdpa', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`BDPA attention kernel "${variant}" requires unsupported GPU features.`);
  }
  const maxKVLen = config.variantMetadata?.maxKVLen;
  if (Number.isFinite(maxKVLen) && kvLen > maxKVLen) {
    throw new Error(`BDPA attention requires kvLen <= ${maxKVLen} but got ${kvLen}.`);
  }

  const kernel = new AttentionBDPAKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_bdpa variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_bdpa_output');

  const uniformBuffer = createBDPAAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
  });

  assertAttentionBindGroupBuffer('attention_bdpa', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 2, 'basisK', basisK?.buffer, [
    `basisKLabel=${basisK?.label ?? 'unknown'}`,
    `basisKDtype=${basisK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 3, 'basisV', basisV?.buffer, [
    `basisVLabel=${basisV?.label ?? 'unknown'}`,
    `basisVDtype=${basisV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 4, 'pagedK', pagedK);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 5, 'pagedV', pagedV);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 6, 'index', index);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 7, 'ropeCos', ropeCos);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 8, 'ropeSin', ropeSin);
  assertAttentionBindGroupBuffer('attention_bdpa', variant, 9, 'output', outputBuf);

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_bdpa_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: basisK.buffer } },
      { binding: 3, resource: { buffer: basisV.buffer } },
      { binding: 4, resource: { buffer: pagedK } },
      { binding: 5, resource: { buffer: pagedV } },
      { binding: 6, resource: { buffer: index } },
      { binding: 7, resource: { buffer: ropeCos } },
      { binding: 8, resource: { buffer: ropeSin } },
      { binding: 9, resource: { buffer: outputBuf } },
    ],
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_bdpa_output');
}

async function executeAttention(
  recorder,
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    bidirectionalSpanStart = 0,
    bidirectionalSpanLength = 0,
    startPos = 0,
    layerIdx,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    indirectBuffer = null,
    indirectOffset = 0,
    kvStart = 0,
    kvLayout = 'contiguous',
    kvPageTable = null,
    kvPageSize = 0,
    kernelPath = null,
  } = options;
  if (!Number.isFinite(bidirectionalSpanStart) || Math.floor(bidirectionalSpanStart) !== bidirectionalSpanStart || bidirectionalSpanStart < 0) {
    throw new Error(`Attention bidirectionalSpanStart must be a non-negative integer, got ${bidirectionalSpanStart}.`);
  }
  if (!Number.isFinite(bidirectionalSpanLength) || Math.floor(bidirectionalSpanLength) !== bidirectionalSpanLength || bidirectionalSpanLength < 0) {
    throw new Error(`Attention bidirectionalSpanLength must be a non-negative integer, got ${bidirectionalSpanLength}.`);
  }
  if (bidirectionalSpanLength > 0 && (bidirectionalSpanStart + bidirectionalSpanLength) > (kvStart + kvLen)) {
    throw new Error(
      `Attention bidirectional span [${bidirectionalSpanStart}, ${bidirectionalSpanStart + bidirectionalSpanLength}) ` +
      `exceeds KV extent [${kvStart}, ${kvStart + kvLen}).`
    );
  }

  // Flash-attention prefill path: raises RDNA3 occupancy via KV-axis workgroup
  // splitting + online-softmax reduction. Gated by options.useFlashPrefill so
  // callers opt in deliberately (runtime config flag). Conservative conditions:
  // head_dim=256, prefill (seqLen>1), contiguous KV, no bidirectional span.
  if (
    options.useFlashPrefill === true
    && headDim === FLASH_HEAD_DIM
    && seqLen > 1
    && kvLayout === 'contiguous'
    && bidirectionalSpanLength === 0
    && indirectBuffer == null
    && K?.dtype === 'f16'
    && V?.dtype === 'f16'
  ) {
    return executeFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, {
      seqLen,
      kvLen,
      numKVHeads,
      scale,
      causal,
      startPos,
      outputBuffer,
      attnSoftcap,
      slidingWindow,
      kvLenBuffer,
      kvStart,
      kvLayout,
      kvPageTable,
      kvPageSize,
    });
  }

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();

  const kvDtype = K.dtype;
  const qDtype = Q.dtype;
  const isPaged = kvLayout === 'paged';
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    qDtype,
    sharedLimit,
    caps,
    layerIdx,
    isPaged,
    kernelPath
  );

  if (execution.recorder) {
    trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);
  }

  const kernel = new AttentionKernel(execution.device);
  const pipeline = await kernel.getPipeline(plan.variant);

  const outputConfig = getKernelConfig('attention', plan.variant);
  const outputDtype = outputConfig.outputDtype;
  if (!outputDtype) {
    if (execution.recorder) {
      throw new Error(`Kernel config missing outputDtype for attention variant "${plan.variant}".`);
    }
    throw new Error(`[Attention] outputDtype is required for variant "${plan.variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  const uniformBuffer = createAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0,
    kvStart,
    pageSize: kvPageSize,
    kvLayout: kvLayout === 'paged' ? 2 : (kvLayout === 'ring' ? 1 : 0),
    bidirectionalSpanStart,
    bidirectionalSpanLength,
  });

  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(execution.device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(execution.device);
  assertAttentionBindGroupBuffer('attention', plan.variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention', plan.variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 2, 'K', K?.buffer, [
    `KLabel=${K?.label ?? 'unknown'}`,
    `KDtype=${K?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 3, 'V', V?.buffer, [
    `VLabel=${V?.label ?? 'unknown'}`,
    `VDtype=${V?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention', plan.variant, 4, 'output', outputBuf);
  assertAttentionBindGroupBuffer('attention', plan.variant, 5, 'kvLen', kvLenBinding);
  assertAttentionBindGroupBuffer('attention', plan.variant, 6, 'pageTable', pageTableBinding, [
    `kvLayout=${kvLayout}`,
  ]);
  const bindGroup = execution.device.createBindGroup({
    label: 'attention_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: kvLenBinding } },
      { binding: 6, resource: { buffer: pageTableBinding } },
    ],
  });

  if (!indirectBuffer && limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ` +
      `${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }

  if (indirectBuffer) {
    if (execution.recorder) {
      recordDispatchIndirect(execution.recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
    } else {
      dispatchIndirect(execution.device, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
    }
  } else {
    dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, plan.workgroups);
  }

  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_output');
}

// -----------------------------------------------------------------------------
// Flash-attention prefill path (head_dim = 256, f16 KV)
// -----------------------------------------------------------------------------
// Two-pass kernel to raise RDNA3 occupancy: pass 1 processes one KV slice per
// workgroup and writes per-split (acc, m, l) partials; pass 2 merges across
// splits with online softmax. Single recorder, two dispatches — queue order
// handles the read-after-write between passes.

const FLASH_BLOCK_SIZE = 32;
const FLASH_HEAD_DIM = 256;
const FLASH_HEAD_DIM_VECS = 64;
const FLASH_REDUCE_WG = 64;

let flashPrefillKernel = null;
let flashReduceKernel = null;

class FlashAttentionPrefillKernel extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor('attention', variant);
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }
}

function getFlashPrefillKernel(device) {
  if (!flashPrefillKernel) {
    flashPrefillKernel = new FlashAttentionPrefillKernel(device);
  }
  return flashPrefillKernel;
}

function getFlashReduceKernel(device) {
  if (!flashReduceKernel) {
    flashReduceKernel = new FlashAttentionPrefillKernel(device);
  }
  return flashReduceKernel;
}

function createFlashAttentionUniformBuffer(device, recorder, params) {
  // Layout mirrors the flash kernel's Uniforms struct (see
  // attention_prefill_flash_head256_f16kv.wgsl). 64 bytes total.
  return createUniformBufferWithView(
    'attention_flash_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      view.setUint32(40, params.kvLenSource, true);
      view.setUint32(44, params.kvStart ?? 0, true);
      view.setUint32(48, params.pageSize ?? 0, true);
      view.setUint32(52, params.kvLayout ?? 0, true);
      view.setUint32(56, params.numKvSplits, true);
      view.setUint32(60, 0, true);
    },
    recorder,
    device
  );
}

function createFlashReduceUniformBuffer(device, recorder, params) {
  return createUniformBufferWithView(
    'attention_flash_reduce_uniforms',
    16,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.queryLen, true);
      view.setUint32(8, params.numKvSplits, true);
      view.setUint32(12, 0, true);
    },
    recorder,
    device
  );
}

function chooseFlashNumKvSplits(kvLen) {
  // Target roughly 32 workgroups × num_heads × num_kv_splits ≈ 4x RDNA3
  // compute-unit count. Keep at least 2× FLASH_BLOCK_SIZE KV positions per
  // split so each workgroup has enough work to amortise dispatch overhead.
  // Short prefills (kvLen ≤ 2 × BLOCK_SIZE) take the single-split fast path
  // which skips the reduce pass entirely.
  if (kvLen <= 2 * FLASH_BLOCK_SIZE) return 1;
  const maxSplits = Math.min(8, Math.floor(kvLen / (2 * FLASH_BLOCK_SIZE)));
  return Math.max(1, maxSplits);
}

export async function executeFlashAttentionPrefill(recorder, Q, K, V, numHeads, headDim, options = {}) {
  if (headDim !== FLASH_HEAD_DIM) {
    throw new Error(`[FlashAttention] headDim must be ${FLASH_HEAD_DIM}, got ${headDim}.`);
  }
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    kvStart = 0,
    kvLayout = 'contiguous',
    kvPageTable = null,
    kvPageSize = 0,
  } = options;

  if (kvLayout !== 'contiguous') {
    throw new Error(`[FlashAttention] kvLayout must be "contiguous", got "${kvLayout}".`);
  }

  const device = execution.device;
  const numQueryBlocks = Math.max(1, Math.ceil(seqLen / FLASH_BLOCK_SIZE));
  const numKvSplits = chooseFlashNumKvSplits(kvLen);
  const singleSplit = numKvSplits === 1;

  // Final output buffer (always allocated; bound to slot 4 on the single-split
  // fast path where the kernel writes normalised output directly).
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_flash_output');

  // Intermediate buffers for the multi-split path. On the single-split fast
  // path we bypass the reduce pass entirely and bind the output buffer in
  // slot 4 (partial_acc slot) while m/l bindings get tiny stub buffers — the
  // kernel skips writes to them.
  const partialAccBytes = singleSplit
    ? 4
    : numQueryBlocks * numHeads * numKvSplits * FLASH_BLOCK_SIZE * FLASH_HEAD_DIM * 4;
  const partialStatsBytes = singleSplit
    ? 4
    : numQueryBlocks * numHeads * numKvSplits * FLASH_BLOCK_SIZE * 4;
  const partialAcc = singleSplit
    ? outputBuf
    : acquireBuffer(partialAccBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'flash_partial_acc');
  const partialM = acquireBuffer(
    partialStatsBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'flash_partial_m'
  );
  const partialL = acquireBuffer(
    partialStatsBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'flash_partial_l'
  );

  // Pass 1 uniforms + dispatch.
  const flashUniform = createFlashAttentionUniformBuffer(device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0,
    kvStart,
    pageSize: kvPageSize,
    kvLayout: 0, // contiguous only for now
    numKvSplits,
  });

  const flashKernel = getFlashPrefillKernel(device);
  const flashPipeline = await flashKernel.getPipeline('prefill_flash_head256_f16kv');
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(device);

  const flashBindGroup = device.createBindGroup({
    label: 'attention_flash_prefill_bg',
    layout: flashPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: flashUniform } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: partialAcc } },
      { binding: 5, resource: { buffer: partialM } },
      { binding: 6, resource: { buffer: partialL } },
      { binding: 7, resource: { buffer: kvLenBinding } },
      { binding: 8, resource: { buffer: pageTableBinding } },
    ],
  });

  dispatchAttentionKernel(
    execution,
    flashKernel,
    flashPipeline,
    flashBindGroup,
    numQueryBlocks * numHeads * numKvSplits
  );
  releaseAttentionUniform(execution, flashUniform);

  // Pass 2 — reduce. Skipped on the single-split fast path where pass 1
  // already wrote the final normalised output directly to outputBuf.
  if (!singleSplit) {
    const reduceUniform = createFlashReduceUniformBuffer(device, execution.recorder, {
      numHeads,
      queryLen: seqLen,
      numKvSplits,
    });

    const reduceKernel = getFlashReduceKernel(device);
    const reducePipeline = await reduceKernel.getPipeline('prefill_flash_reduce');

    const reduceBindGroup = device.createBindGroup({
      label: 'attention_flash_reduce_bg',
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: reduceUniform } },
        { binding: 1, resource: { buffer: partialAcc } },
        { binding: 2, resource: { buffer: partialM } },
        { binding: 3, resource: { buffer: partialL } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const totalQh = seqLen * numHeads;
    const reduceWgX = Math.ceil(totalQh / FLASH_REDUCE_WG);
    dispatchAttentionKernel(
      execution,
      reduceKernel,
      reducePipeline,
      reduceBindGroup,
      [reduceWgX, FLASH_HEAD_DIM_VECS, 1]
    );
    releaseAttentionUniform(execution, reduceUniform);
  }

  // Release intermediate buffers via the recorder's deferred cleanup so GPU
  // work completes before they re-enter the pool. On the single-split path
  // partialAcc IS the output buffer, so we skip it.
  const intermediates = singleSplit ? [partialM, partialL] : [partialAcc, partialM, partialL];
  if (execution.recorder) {
    for (const buf of intermediates) {
      execution.recorder.trackTemporaryBuffer(buf);
    }
  } else {
    device.queue.onSubmittedWorkDone().then(() => {
      for (const buf of intermediates) {
        releaseBuffer(buf);
      }
    });
  }

  return createTensor(outputBuf, 'f32', [seqLen, numHeads, headDim], 'attention_flash_output');
}

async function executeAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    coldLen = 0,
    hotLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    hotWindow = hotLen,
    hotStart = 0,
    coldPageTable = null,
    coldPageSize = 0,
    coldLayout = 2,
    hotLayout = 1,
  } = options;

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }

  const useF16 = Q.dtype === 'f16' && hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const useF16KV = hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const variant = selectKernelRuleValue('attention', 'tieredVariant', { useF16 });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered attention kernel "${variant}" requires unsupported GPU features.`);
  }
  if (!useF16KV) {
    throw new Error('Tiered attention requires f16 KV buffers.');
  }

  const kernel = new AttentionTieredKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_output');

  const uniformBuffer = createTieredAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    coldLen,
    hotLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    hotWindow,
    hotStart,
    coldPageSize,
    coldLayout,
    hotLayout,
  });

  const pageTableBinding = coldPageTable || getPageTableFallbackBuffer(execution.device);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 4, 'coldK', coldK?.buffer, [
    `coldKLabel=${coldK?.label ?? 'unknown'}`,
    `coldKDtype=${coldK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 5, 'coldV', coldV?.buffer, [
    `coldVLabel=${coldV?.label ?? 'unknown'}`,
    `coldVDtype=${coldV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 6, 'output', outputBuf);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 7, 'pageTable', pageTableBinding, [
    `coldLayout=${coldLayout}`,
  ]);
  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldK.buffer } },
      { binding: 5, resource: { buffer: coldV.buffer } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: pageTableBinding } },
    ],
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_output');
}

async function executeAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    coldLen = 0,
    hotLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    hotWindow = hotLen,
    hotStart = 0,
    packedStride = 0,
    mode = 'int8',
    // TurboQuant additional buffers
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
  } = options;

  if (mode === 'turboquant_outlier') {
    throw new Error(
      'TurboQuant outlier attention is not supported yet. ' +
      'Outlier-mode decode kernels are not wired end to end.'
    );
  }

  const isTurboQuant = mode === 'turboquant' || mode === 'turboquant_prod';
  const isProd = mode === 'turboquant_prod';

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredQuantMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered quant attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Tiered quant attention requires packedStride > 0.');
  }

  if (Q.dtype !== 'f32') {
    throw new Error('Tiered quant attention requires f32 Q.');
  }

  if (isTurboQuant && !rotationMatrixBuffer) {
    throw new Error('TurboQuant tiered quant attention requires rotationMatrixBuffer.');
  }
  if (isTurboQuant && !codebookCentroidsBuffer) {
    throw new Error('TurboQuant tiered quant attention requires codebookCentroidsBuffer.');
  }
  if (isProd && !qjlMatrixBuffer) {
    throw new Error('TurboQuant prod tiered quant attention requires qjlMatrixBuffer.');
  }

  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionTieredQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_quant_output');

  const uniformBuffer = createTieredQuantAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    coldLen,
    hotLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    hotWindow,
    hotStart,
    packedStride,
  });

  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 4, 'coldPackedK', coldPackedK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 5, 'coldPackedV', coldPackedV);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 6, 'coldScalesK', coldScalesK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 7, 'coldScalesV', coldScalesV);

  let entries;
  if (isProd) {
    // TurboQuant prod tiered: 16 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 15, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: residualKBuffer } },
      { binding: 9, resource: { buffer: residualVBuffer } },
      { binding: 10, resource: { buffer: residualNormsKBuffer } },
      { binding: 11, resource: { buffer: residualNormsVBuffer } },
      { binding: 12, resource: { buffer: outputBuf } },
      { binding: 13, resource: { buffer: rotationMatrixBuffer } },
      { binding: 14, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 15, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else if (isTurboQuant) {
    // TurboQuant MSE tiered: 11 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
      { binding: 9, resource: { buffer: rotationMatrixBuffer } },
      { binding: 10, resource: { buffer: codebookCentroidsBuffer } },
    ];
  } else {
    // Standard int4/int8: 9 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'output', outputBuf);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_quant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_quant_output');
}

export async function runAttentionBDPA(
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(null, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function recordAttentionBDPA(
  recorder,
  Q,
  basisK,
  basisV,
  pagedK,
  pagedV,
  index,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionBDPA(recorder, Q, basisK, basisV, pagedK, pagedV, index, numHeads, headDim, options);
}

export async function runAttention(
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(null, Q, K, V, mask, numHeads, headDim, options);
}

export async function recordAttention(
  recorder,
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttention(recorder, Q, K, V, mask, numHeads, headDim, options);
}

export async function runAttentionTiered(
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(null, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}

export async function recordAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(recorder, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}

export async function runAttentionTieredQuant(
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    null,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}

export async function recordAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    recorder,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}


// =============================================================================
// Contiguous Quantized Attention (TurboQuant for full-attention models)
// =============================================================================

async function executeAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    packedStride = 0,
    mode = 'turboquant',
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    // Prod-mode additional buffers
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
    packedStrideMSE = 0,
    packedStrideResidual = 0,
  } = options;

  const maxKVLen = getContiguousQuantMaxKVLen();
  if (kvLen > maxKVLen) {
    throw new Error(`Contiguous quant attention requires kvLen <= ${maxKVLen} but got ${kvLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Contiguous quant attention requires packedStride > 0.');
  }
  if (Q.dtype !== 'f32') {
    throw new Error('Contiguous quant attention requires f32 Q.');
  }
  if (!rotationMatrixBuffer) {
    throw new Error('Contiguous quant attention requires rotationMatrixBuffer.');
  }
  if (!codebookCentroidsBuffer) {
    throw new Error('Contiguous quant attention requires codebookCentroidsBuffer.');
  }

  const isProd = mode === 'turboquant_prod';
  const variant = selectKernelRuleValue('attention', 'contiguousQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_contiguous_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Contiguous quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionContiguousQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_contiguous_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_contiguous_quant_output');

  const uniformParams = isProd
    ? {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStrideMSE, packedStrideResidual,
    }
    : {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStride,
    };
  const uniformBuffer = createContiguousQuantAttentionUniformBuffer(
    execution.device, execution.recorder, uniformParams
  );

  let entries;
  if (isProd) {
    // Contiguous prod: 14 bindings
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 6, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 7, 'residual_v', residualVBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 13, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: residualKBuffer } },
      { binding: 7, resource: { buffer: residualVBuffer } },
      { binding: 8, resource: { buffer: residualNormsKBuffer } },
      { binding: 9, resource: { buffer: residualNormsVBuffer } },
      { binding: 10, resource: { buffer: outputBuf } },
      { binding: 11, resource: { buffer: rotationMatrixBuffer } },
      { binding: 12, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 13, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else {
    // Contiguous MSE: 9 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: rotationMatrixBuffer } },
      { binding: 8, resource: { buffer: codebookCentroidsBuffer } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_contiguous_quant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_contiguous_quant_output');
}

export async function runAttentionContiguousQuant(
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    null, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}

export async function recordAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    recorder, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}
