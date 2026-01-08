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

import { getDevice, getDeviceLimits, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { DIMENSION_LIMITS, MEMORY_THRESHOLDS, TILE_SIZES } from './constants.js';
import { getKernelThresholds } from '../../config/schema/kernel-thresholds.schema.js';
import { createUniformBufferWithView, getKernelConfig, hasRequiredFeatures } from './utils.js';
import { dispatchIndirect, recordDispatchIndirect } from './dispatch.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { log, trace } from '../../debug/index.js';
import { getKernelPathAttentionVariant, getKernelPathStrict } from '../../config/kernel-path-loader.js';

// Track if we've logged the attention tier selection (avoid spam)
let loggedAttentionTier = false;

/**
 * Get max KV length for chunked attention from kernel config.
 * Cached for performance since it's called frequently.
 * @type {number | null}
 */
let _chunkedMaxKVLen = null;

/**
 * @returns {number}
 */
function getChunkedMaxKVLen() {
  if (_chunkedMaxKVLen === null) {
    const config = getKernelConfig('attention', 'decode_chunked_f16kv');
    _chunkedMaxKVLen = config.variantMetadata?.maxKVLen ?? 2048;
  }
  return _chunkedMaxKVLen;
}

/** @type {GPUBuffer | null} */
let kvLenFallbackBuffer = null;

/**
 * @param {GPUDevice} device
 * @returns {GPUBuffer}
 */
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

/**
 * @typedef {'subgroup' | 'tiled_large' | 'tiled_small' | 'streaming'} AttentionTier
 */

/**
 * @typedef {Object} AttentionPlan
 * @property {AttentionTier} tier
 * @property {string} variant
 * @property {number} workgroups
 * @property {boolean} useF16KV
 * @property {boolean} isDecode
 */

class AttentionKernel extends KernelBase {
  /**
   * @param {string} variant
   * @returns {Promise<GPUComputePipeline>}
   */
  async getPipeline(variant) {
    return this.getPipelineFor('attention', variant);
  }

  /**
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number} workgroups
   */
  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }

  /**
   * @param {import('../command-recorder.js').CommandRecorder} recorder
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number} workgroups
   */
  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
}

/**
 * @param {number} headDim
 * @param {number} seqLen
 * @param {boolean} useF16KV
 * @param {AttentionTier | null} forcedTier
 * @param {number} sharedLimit
 * @param {ReturnType<typeof getKernelCapabilities>} caps
 * @param {boolean} strict
 * @returns {AttentionTier}
 */
function selectAttentionTier(
  headDim,
  seqLen,
  useF16KV,
  forcedTier,
  sharedLimit,
  caps,
  strict
) {
  const isDecode = seqLen === 1;
  const canLarge =
    headDim <= DIMENSION_LIMITS.ATTENTION_LARGE_MAX_HEAD_DIM &&
    sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_LARGE_SHARED;
  const smallRequired = useF16KV
    ? MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F16
    : MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F32;
  const canSmall =
    headDim <= DIMENSION_LIMITS.ATTENTION_SMALL_MAX_HEAD_DIM &&
    sharedLimit >= smallRequired;
  const canSubgroup =
    caps.hasSubgroups &&
    headDim <= DIMENSION_LIMITS.ATTENTION_SUBGROUP_MAX_HEAD_DIM &&
    sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_SUBGROUP_SHARED &&
    isDecode;

  /**
   * @param {string} message
   */
  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    log.warn('Attention', message);
  };

  let tier = forcedTier;

  if (tier === 'tiled_large' && !canLarge) {
    failOrWarn(`Requested tiled_large but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
    tier = null;
  }
  if (tier === 'tiled_small' && !canSmall) {
    failOrWarn(`Requested tiled_small but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
    tier = null;
  }
  if (tier === 'subgroup' && !canSubgroup) {
    failOrWarn(`Requested subgroup attention but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}, subgroups=${caps.hasSubgroups}).`);
    tier = null;
  }

  if (!tier) {
    if (canSubgroup) {
      tier = 'subgroup';
      if (!loggedAttentionTier) {
        trace.attn(0, `Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
        loggedAttentionTier = true;
      }
    } else if (canLarge) {
      tier = 'tiled_large';
    } else if (canSmall) {
      tier = 'tiled_small';
    } else if (isDecode) {
      tier = 'streaming';
    } else {
      log.warn('Attention', `No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). Falling back to streaming. Expect slow prefill.`);
      tier = 'streaming';
    }
  }

  return /** @type {AttentionTier} */ (tier);
}

// Track if we've logged chunked kernel selection
let loggedChunkedKernel = false;

/**
 * @param {AttentionTier} tier
 * @param {boolean} isDecode
 * @param {boolean} useF16KV
 * @param {number} numHeads
 * @param {number} headDim
 * @param {number} kvLen
 * @returns {string}
 */
function resolveAttentionVariant(
  tier,
  isDecode,
  useF16KV,
  numHeads,
  headDim,
  kvLen
) {
  const base = isDecode ? 'decode' : 'prefill';

  // Check if chunked kernel is viable:
  // - Decode only (seqLen=1)
  // - F16 KV cache
  // - Large headDim (parallelizes across dimensions)
  // - KV length within shared memory limit (from kernel config)
  const chunkedMaxKVLen = getChunkedMaxKVLen();
  const minHeadDimForChunked = getKernelThresholds().attention.minHeadDimForChunked;
  const canUseChunked = isDecode && useF16KV && headDim >= minHeadDimForChunked && kvLen <= chunkedMaxKVLen;
  const decodeSubgroupMaxKVLen = chunkedMaxKVLen;
  const decodeSubgroupMaxHeadDim = getKernelThresholds().attention.tierHeadDimLimits.tier1;
  const canUseDecodeSubgroup = isDecode && !useF16KV && headDim <= decodeSubgroupMaxHeadDim && kvLen <= decodeSubgroupMaxKVLen;

  if (tier === 'subgroup') {
    // decode_subgroup only supports F32 KV cache
    // Fall back to chunked for F16 KV cache (much faster than streaming)
    if (useF16KV) {
      if (canUseChunked) {
        if (!loggedChunkedKernel) {
          trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
          loggedChunkedKernel = true;
        }
        return 'decode_chunked_f16kv';
      }
      return 'decode_streaming_f16kv';
    }
    if (canUseDecodeSubgroup) {
      return 'decode_subgroup';
    }
    return 'decode_streaming';
  }
  if (tier === 'tiled_large') {
    return base + (useF16KV ? '_f16kv' : '');
  }
  if (tier === 'tiled_small') {
    return `${base}_small${useF16KV ? '_f16kv' : ''}`;
  }
  // For streaming tier, prefer chunked if viable
  if (canUseChunked) {
    if (!loggedChunkedKernel) {
      trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
      loggedChunkedKernel = true;
    }
    return 'decode_chunked_f16kv';
  }
  return `${base}_streaming${useF16KV ? '_f16kv' : ''}`;
}

/**
 * @param {AttentionTier} tier
 * @param {number} seqLen
 * @param {number} numHeads
 * @returns {number}
 */
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

/**
 * @param {string} variant
 * @returns {AttentionTier}
 */
function inferAttentionTierFromVariant(variant) {
  if (variant === 'decode_subgroup') return 'subgroup';
  if (variant.startsWith('prefill_streaming') || variant.startsWith('decode_streaming') || variant === 'decode_chunked_f16kv') {
    return 'streaming';
  }
  if (variant.startsWith('prefill_small') || variant.startsWith('decode_small')) return 'tiled_small';
  return 'tiled_large';
}

/**
 * @param {string} variant
 * @param {boolean} isDecode
 * @param {boolean} useF16KV
 * @param {ReturnType<typeof getKernelCapabilities>} caps
 * @param {boolean} strict
 * @returns {string | null}
 */
function validateAttentionVariant(
  variant,
  isDecode,
  useF16KV,
  caps,
  strict
) {
  const normalized = variant.trim();
  /**
   * @param {string} message
   * @returns {string | null}
   */
  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    log.warn('Attention', message);
    return null;
  };

  let config;
  try {
    config = getKernelConfig('attention', normalized);
  } catch {
    return failOrWarn(`Unknown attention kernel variant "${variant}".`);
  }

  if (!hasRequiredFeatures(config.requires, caps)) {
    return failOrWarn(`Attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const expectsF16KV = normalized.includes('_f16kv');
  if (expectsF16KV !== useF16KV) {
    const kvLabel = useF16KV ? 'f16' : 'f32';
    return failOrWarn(`Attention kernel "${variant}" incompatible with ${kvLabel} KV cache.`);
  }

  const isDecodeVariant = normalized.startsWith('decode');
  const isPrefillVariant = normalized.startsWith('prefill');
  if (isDecode && isPrefillVariant) {
    return failOrWarn(`Attention kernel "${variant}" is prefill-only but decode requested.`);
  }
  if (!isDecode && isDecodeVariant) {
    return failOrWarn(`Attention kernel "${variant}" is decode-only but prefill requested.`);
  }

  return normalized;
}

/**
 * @param {number} seqLen
 * @param {number} kvLen
 * @param {number} headDim
 * @param {number} numHeads
 * @param {string} kvDtype
 * @param {number} sharedLimit
 * @param {ReturnType<typeof getKernelCapabilities>} caps
 * @param {number} [layerIdx]
 * @returns {AttentionPlan}
 */
function resolveAttentionPlan(
  seqLen,
  kvLen,
  headDim,
  numHeads,
  kvDtype,
  sharedLimit,
  caps,
  layerIdx
) {
  const useF16KV = kvDtype === 'f16';
  const isDecode = seqLen === 1;
  const strict = getKernelPathStrict();
  const pathVariant = getKernelPathAttentionVariant(isDecode ? 'decode' : 'prefill', layerIdx);

  if (pathVariant) {
    const variantOverride = validateAttentionVariant(pathVariant, isDecode, useF16KV, caps, strict);
    if (variantOverride) {
      const tier = inferAttentionTierFromVariant(variantOverride);
      const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);
      return { tier, variant: variantOverride, workgroups, useF16KV, isDecode };
    }
  }

  const tier = selectAttentionTier(headDim, seqLen, useF16KV, null, sharedLimit, caps, strict);
  const variant = resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim, kvLen);
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);

  return { tier, variant, workgroups, useF16KV, isDecode };
}

/**
 * @param {GPUDevice} device
 * @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {Object} params
 * @param {number} params.numHeads
 * @param {number} params.numKVHeads
 * @param {number} params.headDim
 * @param {number} params.kvLen
 * @param {number} params.seqLen
 * @param {number} params.scale
 * @param {boolean} params.causal
 * @param {number} params.startPos
 * @param {number} params.attnSoftcap
 * @param {number} params.slidingWindow
 * @param {number} params.kvLenSource
 * @returns {GPUBuffer}
 */
function createAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_uniforms',
    48, // 44 bytes used + 4 padding for 16-byte alignment
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
    },
    recorder,
    device
  );
}

/**
 * Run attention operation
 * @param {import('../tensor.js').Tensor} Q
 * @param {import('../tensor.js').Tensor} K
 * @param {import('../tensor.js').Tensor} V
 * @param {GPUBuffer | null} mask
 * @param {number} numHeads
 * @param {number} headDim
 * @param {import('./attention.js').AttentionOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
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
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  /** @type {import('../tensor.js').TensorDtype} */
  const kvDtype = K.dtype;
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    sharedLimit,
    caps,
    layerIdx
  );
  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  // Output is always f32 (attention scores computed in f32)
  /** @type {import('../tensor.js').TensorDtype} */
  const outputDtype = 'f32';
  const outputSize = seqLen * numHeads * headDim * 4;
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
  });

  // Create bind group
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
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

/**
 * Record attention operation (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} Q
 * @param {import('../tensor.js').Tensor} K
 * @param {import('../tensor.js').Tensor} V
 * @param {GPUBuffer | null} mask
 * @param {number} numHeads
 * @param {number} headDim
 * @param {import('./attention.js').AttentionOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
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
  } = options;

  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  /** @type {import('../tensor.js').TensorDtype} */
  const kvDtype = K.dtype;
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    sharedLimit,
    caps,
    layerIdx
  );

  trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);

  const kernel = new AttentionKernel(device);
  const pipeline = await kernel.getPipeline(plan.variant);

  // Output is always f32 (attention scores computed in f32)
  /** @type {import('../tensor.js').TensorDtype} */
  const outputDtype = 'f32';
  const outputSize = seqLen * numHeads * headDim * 4;
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
  });

  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device);
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
