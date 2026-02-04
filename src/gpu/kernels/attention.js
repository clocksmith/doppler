

import { getDevice, getDeviceLimits, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { TILE_SIZES } from './constants.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { createUniformBufferWithView, getKernelConfig, hasRequiredFeatures } from './utils.js';
import { dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { trace } from '../../debug/index.js';
import { getKernelPathAttentionVariant } from '../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../kernel-selection-log.js';

// Track if we've logged the attention tier selection (avoid spam)
let loggedAttentionTier = false;


let _chunkedMaxKVLen = null;


function getChunkedMaxKVLen() {
  if (_chunkedMaxKVLen === null) {
    const config = getKernelConfig('attention', 'decode_chunked_f16kv');
    const maxKVLen = config.variantMetadata?.maxKVLen;
    if (!Number.isFinite(maxKVLen)) {
      throw new Error('Kernel config missing attention.decode_chunked_f16kv maxKVLen');
    }
    _chunkedMaxKVLen = maxKVLen;
  }
  return _chunkedMaxKVLen;
}

let _tieredMaxKVLen = null;

function getTieredMaxKVLen() {
  if (_tieredMaxKVLen === null) {
    const config = getKernelConfig('attention_tiered', 'decode_tiered_f16');
    const maxKVLen = config.variantMetadata?.maxKVLen;
    if (!Number.isFinite(maxKVLen)) {
      throw new Error('Kernel config missing attention_tiered.decode_tiered_f16 maxKVLen');
    }
    _tieredMaxKVLen = maxKVLen;
  }
  return _tieredMaxKVLen;
}

let _tieredQuantMaxKVLen = null;

function getTieredQuantMaxKVLen() {
  if (_tieredQuantMaxKVLen === null) {
    const config = getKernelConfig('attention_tiered_quant', 'decode_tiered_int8_f16kv');
    const maxKVLen = config.variantMetadata?.maxKVLen;
    if (!Number.isFinite(maxKVLen)) {
      throw new Error('Kernel config missing attention_tiered_quant.decode_tiered_int8_f16kv maxKVLen');
    }
    _tieredQuantMaxKVLen = maxKVLen;
  }
  return _tieredQuantMaxKVLen;
}


let kvLenFallbackBuffer = null;


function getKvLenFallbackBuffer(device) {
  if (!kvLenFallbackBuffer) {
    kvLenFallbackBuffer = device.createBuffer({
      label: 'attention_kv_len_fallback',
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(kvLenFallbackBuffer, 0, new Uint32Array([0]));
  }
  return kvLenFallbackBuffer;
}

let pageTableFallbackBuffer = null;

function getPageTableFallbackBuffer(device) {
  if (!pageTableFallbackBuffer) {
    pageTableFallbackBuffer = device.createBuffer({
      label: 'attention_page_table_fallback',
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pageTableFallbackBuffer, 0, new Uint32Array([0]));
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
  if (variant === 'decode_subgroup') return 'subgroup';
  if (variant.startsWith('decode_online')) return 'subgroup';
  if (variant.startsWith('decode_paged')) return 'tiled_large';
  if (variant.startsWith('prefill_streaming') || variant.startsWith('decode_streaming') || variant === 'decode_chunked_f16kv') {
    return 'streaming';
  }
  if (variant.startsWith('prefill_small') || variant.startsWith('decode_small')) return 'tiled_small';
  return 'tiled_large';
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
    if (!caps.hasSubgroups) {
      throw new Error(`Attention kernel "${variant}" requires subgroup support.`);
    }
    if (headDim > thresholds.subgroupMaxHeadDim) {
      throw new Error(`Attention kernel "${variant}" requires headDim <= ${thresholds.subgroupMaxHeadDim} but got ${headDim}.`);
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
      const requiredShared = isSmall
        ? (useF16KV ? thresholds.smallSharedF16 : thresholds.smallSharedF32)
        : (useF16KV ? thresholds.largeSharedF16 : thresholds.largeSharedF32);
      const maxHeadDim = isSmall ? thresholds.smallMaxHeadDim : thresholds.largeMaxHeadDim;
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
  isPaged
) {
  const useF16KV = kvDtype === 'f16';
  const useF16Q = qDtype === 'f16';
  const isDecode = seqLen === 1;
  const phase = selectKernelRuleValue('attention', 'phase', { isDecode });
  const pathVariant = getKernelPathAttentionVariant(phase, layerIdx);

  if (pathVariant) {
    const variantOverride = validateAttentionVariant(
      pathVariant,
      isDecode,
      useF16KV,
      useF16Q,
      caps,
      headDim,
      kvLen,
      sharedLimit
    );
    const tier = inferAttentionTierFromVariant(variantOverride);
    const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);
    logKernelSelectionOnce('attention', {
      variant: variantOverride,
      reason: `path_override:${tier}`,
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
  isPaged = false
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
    isPaged
  );
}


function createAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_uniforms',
    64, // 60 bytes used + 4 padding for 16-byte alignment
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
      view.setUint32(56, 0, true);
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


export async function runAttention(
  Q,
  K,
  V,
  mask,
  numHeads,
  headDim,
  options = {}
) {
  const device = getDevice();
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
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
  } = options;

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
    isPaged
  );
  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  const outputConfig = getKernelConfig('attention', plan.variant);
  const outputDtype = outputConfig.outputDtype;
  if (!outputDtype) {
    throw new Error(`[Attention] outputDtype is required for variant "${plan.variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  // Pad output to Q4K alignment (256) so downstream o_proj can read full blocks
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  // Create uniform buffer
  const uniformBuffer = createAttentionUniformBuffer(device, null, {
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
  });

  // Create bind group
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(device);
  const bindGroup = device.createBindGroup({
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
    dispatchIndirect(device, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
  } else {
    kernel.dispatch(pipeline, bindGroup, plan.workgroups);
  }

  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_output');
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
  const device = recorder.device;
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
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
  } = options;

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
    isPaged
  );

  trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);

  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  const outputConfig = getKernelConfig('attention', plan.variant);
  const outputDtype = outputConfig.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention variant "${plan.variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  // Pad output to Q4K alignment (256) so downstream o_proj can read full blocks
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_output');

  const uniformBuffer = createAttentionUniformBuffer(device, recorder, {
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
  });

  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
  const pageTableBinding = kvPageTable || getPageTableFallbackBuffer(device);
  const bindGroup = device.createBindGroup({
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
    recordDispatchIndirect(recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, 'attention');
  } else {
    kernel.record(recorder, pipeline, bindGroup, plan.workgroups);
  }

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_output');
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
  const device = getDevice();
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

  const kernel = new AttentionTieredKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_output');

  const uniformBuffer = createTieredAttentionUniformBuffer(device, null, {
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

  const pageTableBinding = coldPageTable || getPageTableFallbackBuffer(device);
  const bindGroup = device.createBindGroup({
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

  const workgroups = numHeads;
  kernel.dispatch(pipeline, bindGroup, workgroups);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_output');
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
  const device = recorder.device;
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

  const kernel = new AttentionTieredKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_output');

  const uniformBuffer = createTieredAttentionUniformBuffer(device, recorder, {
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

  const pageTableBinding = coldPageTable || getPageTableFallbackBuffer(device);
  const bindGroup = device.createBindGroup({
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

  const workgroups = numHeads;
  kernel.record(recorder, pipeline, bindGroup, workgroups);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_output');
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
  const device = getDevice();
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
  } = options;

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

  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionTieredQuantKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_quant_output');

  const uniformBuffer = createTieredQuantAttentionUniformBuffer(device, null, {
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

  const bindGroup = device.createBindGroup({
    label: 'attention_tiered_quant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numHeads;
  kernel.dispatch(pipeline, bindGroup, workgroups);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_quant_output');
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
  const device = recorder.device;
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
  } = options;

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

  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionTieredQuantKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_quant_output');

  const uniformBuffer = createTieredQuantAttentionUniformBuffer(device, recorder, {
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

  const bindGroup = device.createBindGroup({
    label: 'attention_tiered_quant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numHeads;
  kernel.record(recorder, pipeline, bindGroup, workgroups);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_quant_output');
}
