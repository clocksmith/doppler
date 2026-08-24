import { getDevice, getDeviceEpoch, getDeviceLimits, getKernelCapabilities } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { KernelBase } from '../kernel-base.js';
import { TILE_SIZES } from '../constants.js';
import { getKernelThresholds, padToQ4KBlock } from '../../../config/schema/index.js';
import { createUniformBufferWithView } from '../uniform-utils.js';
import { getKernelConfig } from '../kernel-configs.js';
import { getPipelineBindGroupLayout } from '../pipeline-cache.js';
import { hasRequiredFeatures } from '../feature-check.js';
import { dispatchIndirect, recordDispatchIndirect } from '../dispatch.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';
import { log, trace } from '../../../debug/index.js';
import { getKernelPathAttentionVariant, getKernelPathStrict } from '../../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from '../rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../../kernel-selection-log.js';
import { getRuntimeConfig } from '../../../config/runtime.js';

export let loggedAttentionTier = false;

export function getRequiredVariantMaxKVLen(operation, variant, errorLabel) {
  const config = getKernelConfig(operation, variant);
  const maxKVLen = config.variantMetadata?.maxKVLen;
  if (!Number.isFinite(maxKVLen)) {
    throw new Error(`Kernel config missing ${errorLabel} maxKVLen`);
  }
  return maxKVLen;
}

export function getChunkedMaxKVLen() {
  return getRequiredVariantMaxKVLen('attention', 'decode_chunked_f16kv', 'attention.decode_chunked_f16kv');
}

export function selectAttentionTier(
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

export let loggedChunkedKernel = false;

export function resolveAttentionVariant(
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

export function resolveAttentionQueryBlockSize(tier, variant = null) {
  if (variant) {
    const metadataBlockSize = getKernelConfig('attention', variant).variantMetadata?.queryBlockSize;
    if (metadataBlockSize != null) {
      if (!Number.isInteger(metadataBlockSize) || metadataBlockSize <= 0) {
        throw new Error(`Attention kernel "${variant}" has invalid variantMetadata.queryBlockSize=${metadataBlockSize}.`);
      }
      return metadataBlockSize;
    }
  }
  if (tier === 'tiled_large') {
    return TILE_SIZES.ATTENTION_LARGE_BLOCK_SIZE;
  }
  return TILE_SIZES.ATTENTION_SMALL_BLOCK_SIZE;
}

export function calculateAttentionWorkgroups(tier, seqLen, numHeads, variant = null) {
  if (tier === 'subgroup') {
    return numHeads;
  }
  if (tier === 'streaming') {
    return seqLen * numHeads;
  }
  const queryBlockSize = resolveAttentionQueryBlockSize(tier, variant);
  return Math.ceil(seqLen / queryBlockSize) * numHeads;
}

export function inferAttentionTierFromVariant(variant) {
  const config = getKernelConfig('attention', variant);
  const tier = config.variantMetadata?.tier;
  if (!tier) {
    throw new Error(`Attention kernel "${variant}" missing variantMetadata.tier in registry.`);
  }
  return tier;
}

export function validateAttentionVariant(
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
    const exactHeadDim = config.variantMetadata?.exactHeadDim;
    const maxHeadDim = config.variantMetadata?.maxHeadDim ?? thresholds.subgroupMaxHeadDim;
    if (!caps.hasSubgroups) {
      throw new Error(`Attention kernel "${variant}" requires subgroup support.`);
    }
    if (Number.isFinite(exactHeadDim) && headDim !== exactHeadDim) {
      throw new Error(`Attention kernel "${variant}" requires headDim == ${exactHeadDim} but got ${headDim}.`);
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

export function resolveAttentionPlan(
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
      const workgroups = calculateAttentionWorkgroups(adaptiveSelection.tier, seqLen, numHeads, adaptiveVariant);
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
    const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads, variantOverride);
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
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads, variant);

  logKernelSelectionOnce('attention', {
    variant: validatedVariant,
    reason: selection.reason,
  });

  return { tier, variant: validatedVariant, workgroups, useF16KV, isDecode };
}
