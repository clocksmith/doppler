

import { getDevice, getDeviceLimits, getKernelCapabilities } from '../../../gpu/device.js';
import {
  KVCache,
  SlidingWindowKVCache,
  TieredKVCache,
  BasisDecomposedPagedCache,
  QuantizedKVCache,
  MixedGeometryKVCache,
} from '../../kv-cache.js';
import {
  retainTurboQuantSharedBuffers,
} from '../../../gpu/kernels/turboquant-codebook.js';
import { getKernelConfig } from '../../../gpu/kernels/kernel-configs.js';
import { selectRuleValue as selectKernelRuleValue } from '../../../gpu/kernels/rule-registry.js';
import { Tokenizer } from '../../tokenizer.js';
import { MoERouter } from '../../moe-router.js';
import { SpeculativeDecoder } from '../../speculative.js';
import { getDopplerLoader } from '../../../loader/doppler-loader.js';
import { log, trace as debugTrace } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { PAGED_LAYOUT_SEQ_LEN_THRESHOLD } from '../../../config/schema/index.js';
import {
  kernelPathRequiresF32MatmulWeights,
} from '../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { resolvePerLayerInputsSession } from './generator-helpers.js';
import {
  createHttpArtifactStorageContext,
  createNodeFileArtifactStorageContext,
} from '../../../storage/artifact-storage-context.js';

// Extracted standalone modules
import { initRoPEFrequencies, isGPURoPEBuffers } from './init-rope.js';
import {
  applyChatTemplate,
  applyGemmaChatTemplate,
  applyGemma4ChatTemplate,
  applyLlama3ChatTemplate,
  applyGptOssChatTemplate,
  applyQwenChatTemplate,
  isStopToken,
} from './init-chat-templates.js';
import { fuseQKVWeights } from './init-qkv-fusion.js';

// Re-exports for backwards compatibility
export { initRoPEFrequencies, isGPURoPEBuffers } from './init-rope.js';
export {
  applyChatTemplate,
  applyGemmaChatTemplate,
  applyGemma4ChatTemplate,
  applyLlama3ChatTemplate,
  applyGptOssChatTemplate,
  applyQwenChatTemplate,
  isStopToken,
} from './init-chat-templates.js';
export { fuseQKVWeights } from './init-qkv-fusion.js';

function resolveErrorMessage(error) {
  if (error && typeof error === 'object' && typeof error.message === 'string') {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return String(error);
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRuntimeInferenceConfigInput(value) {
  if (!isPlainObject(value)) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(value, 'session')
    || Object.prototype.hasOwnProperty.call(value, 'kvcache')
    || Object.prototype.hasOwnProperty.call(value, 'compute')
    || Object.prototype.hasOwnProperty.call(value, 'batching')
    || Object.prototype.hasOwnProperty.call(value, 'generation')
    || Object.prototype.hasOwnProperty.call(value, 'kernelPath')
    || Object.prototype.hasOwnProperty.call(value, 'pipeline');
}

function resolveRuntimeKVConfig(runtimeInput) {
  if (runtimeInput != null && !isRuntimeInferenceConfigInput(runtimeInput)) {
    return runtimeInput;
  }
  const runtimeInference = runtimeInput ?? getRuntimeConfig().inference;
  if (runtimeInference?.session?.kvcache) {
    return runtimeInference.session.kvcache;
  }
  throw new Error(
    'runtime.inference.session.kvcache is required for live KV cache creation. ' +
    'Top-level runtime.inference.kvcache is not supported on the live path.'
  );
}

function assertSupportedTurboQuantMode(mode, label) {
  if (String(mode ?? '').trim().toLowerCase() !== 'turboquant_outlier') {
    return;
  }
  throw new Error(
    `${label}="turboquant_outlier" is not supported yet. ` +
    'TurboQuant outlier high-precision buffers and decode kernels are not wired end to end; ' +
    'use "turboquant" or "turboquant_prod".'
  );
}

function toArrayBuffer(value, label) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (value instanceof Uint8Array) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function toUint8Array(value, label) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function isRDRRManifest(manifest) {
  return manifest !== null && typeof manifest === 'object' && Array.isArray( (manifest).shards);
}

export function createNodeFileShardStorageContext(baseUrl, manifest) {
  return createNodeFileArtifactStorageContext(baseUrl, manifest);
}

function createRemoteStorageContext(baseUrl, manifest) {
  const nodeFileStorageContext = createNodeFileArtifactStorageContext(baseUrl, manifest);
  if (nodeFileStorageContext) {
    return nodeFileStorageContext;
  }
  return createHttpArtifactStorageContext(baseUrl, manifest);
}


/** Allowed quantizationInfo.layout values for Q4K models. */
const Q4K_LAYOUT_ALLOWLIST = new Set(['row', 'col']);
const Q4K_PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
  'qkv_proj',
  'linear_qkv_proj',
  'linear_z_proj',
  'linear_a_proj',
  'linear_b_proj',
  'linear_out_proj',
]);

function summarizeQ4KProjectionKernelKinds(kernelPath) {
  const summary = {
    denseProjectionKernels: [],
    fusedProjectionKernels: [],
  };
  if (!kernelPath || typeof kernelPath !== 'object') {
    return summary;
  }

  const appendPhase = (phase, steps) => {
    for (const step of steps ?? []) {
      if (!Q4K_PROJECTION_OPS.has(step?.op)) {
        continue;
      }
      const kernel = String(step?.kernel ?? '');
      if (!kernel) {
        continue;
      }
      const descriptor = `${phase}.${step.op}:${kernel}#${String(step?.entry ?? 'main')}`;
      if (kernel.startsWith('fused_matmul_q4')) {
        summary.fusedProjectionKernels.push(descriptor);
      } else if (kernel.startsWith('matmul_')) {
        summary.denseProjectionKernels.push(descriptor);
      }
    }
  };

  appendPhase('decode', kernelPath.decode?.steps);
  appendPhase('prefill', kernelPath.prefill?.steps);
  return summary;
}

function resolveQ4KProjectionMaterializationMode(
  manifest,
  kernelPath,
  kernelPathSource = 'none'
) {
  const summary = summarizeQ4KProjectionKernelKinds(kernelPath);
  const mode = selectKernelRuleValue('matmul', 'q4kMaterializationMode', {
    hasFusedProjections: summary.fusedProjectionKernels.length > 0,
    hasDenseProjections: summary.denseProjectionKernels.length > 0,
  });
  debugTrace.loader(
    `Q4K materialization: model=${manifest?.modelId ?? 'unknown'}, mode=${mode}, ` +
    `source=${kernelPathSource}, dense=${summary.denseProjectionKernels.length}, ` +
    `fused=${summary.fusedProjectionKernels.length}`
  );
  return mode;
}

export function resolveQ4KConfig(
  manifest,
  kernelPath,
  kernelPathSource = 'none',
  keepF32Weights = false
) {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps != null && caps.hasSubgroups === true;
  // Layout in quantizationInfo: 'row' (fused) or 'col' (dequant)
  const q4kLayout = manifest?.quantizationInfo?.layout ?? null;
  const isQ4KModel = manifest?.quantization === 'Q4_K_M';
  if (isQ4KModel && q4kLayout == null) {
    throw new Error(
      `Manifest "${manifest?.modelId ?? 'unknown'}" is missing quantizationInfo.layout for Q4_K_M. Re-convert the model.`
    );
  }
  if (q4kLayout != null && !Q4K_LAYOUT_ALLOWLIST.has(q4kLayout)) {
    throw new Error(
      `Manifest "${manifest?.modelId ?? 'unknown'}" has invalid quantizationInfo.layout "${q4kLayout}". ` +
      `Allowed values: ${[...Q4K_LAYOUT_ALLOWLIST].join(', ')}.`
    );
  }
  let q4kMaterializationMode = isQ4KModel
    ? resolveQ4KProjectionMaterializationMode(manifest, kernelPath, kernelPathSource)
    : 'dense';
  // Runtime override: retainQ4KMaterialization forces "mixed" mode so the
  // loader keeps the Q4_K packed buffer alongside the dequantized dense
  // buffer. Unlocks `hasQ4KMaterialization=true` in the FFN fusion rule when
  // the execution graph doesn't declare a fused projection kernel.
  if (isQ4KModel && hasSubgroups && q4kMaterializationMode === 'dense') {
    // Manifest-first: inference.session.retainQ4KMaterialization wins when
    // explicitly set (true or false); null/missing falls through to the runtime
    // config so explicit runtime profiles still have last word via the normal merge.
    const manifestRetain = manifest?.inference?.session?.retainQ4KMaterialization;
    const runtimeRetain = (manifestRetain !== null && manifestRetain !== undefined)
      ? manifestRetain === true
      : getRuntimeConfig().inference?.session?.retainQ4KMaterialization === true;
    if (runtimeRetain) {
      q4kMaterializationMode = 'mixed';
      debugTrace.loader(
        `Q4K materialization overridden by runtime flag retainQ4KMaterialization=true: mode=mixed`
      );
    }
  }
  if (isQ4KModel) {
    debugTrace.loader(
      `Q4K projection materialization: model=${manifest?.modelId ?? 'unknown'}, ` +
      `mode=${q4kMaterializationMode}, source=${kernelPathSource}`
    );
  }
  const hasExplicitKernelPath = kernelPath != null;
  let useFused = hasExplicitKernelPath
    ? q4kMaterializationMode !== 'dense'
    : hasSubgroups;
  const kernelPathKeepsF32Weights = kernelPathRequiresF32MatmulWeights(kernelPath);
  if (q4kLayout === 'col') {
    useFused = false;
  }
  const resolvedKeepF32Weights = keepF32Weights || kernelPathKeepsF32Weights;

  const pathLabel = kernelPath?.id ?? 'auto';
  const layoutLabel = q4kLayout ?? 'none';
  debugTrace.loader(
    `Q4K config: fused=${useFused}, kernelPath=${pathLabel}, source=${kernelPathSource}, ` +
    `layout=${layoutLabel}, materialization=${q4kMaterializationMode}, ` +
    `keepF32Weights=${resolvedKeepF32Weights}, subgroups=${hasSubgroups}`
  );

  return {
    useFusedQ4K: useFused,
    q4kLayout,
    keepF32Weights: resolvedKeepF32Weights,
    q4kMaterializationMode,
  };
}

function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

function isSlidingLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding';
}

function hasFullAttentionLayers(layerTypes) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
    return false;
  }
  return layerTypes.some((layerType) => !isSlidingLayerType(layerType));
}

function resolveContiguousKVPolicy(modelConfig) {
  if (hasFullAttentionLayers(modelConfig.layerTypes)) {
    return {
      forceContiguousKVCache: true,
      reason: 'layerPattern',
    };
  }
  if (modelConfig.layerTypes == null && modelConfig.slidingWindow == null) {
    return {
      forceContiguousKVCache: true,
      reason: 'slidingWindow',
    };
  }
  return {
    forceContiguousKVCache: false,
    reason: 'none',
  };
}

function usesMixedGeometryKVCache(modelConfig) {
  if (!modelConfig || typeof modelConfig !== 'object') {
    return false;
  }
  const hasMixedAttentionGeometry = Number.isFinite(modelConfig.globalHeadDim)
    && modelConfig.globalHeadDim > 0
    && modelConfig.globalHeadDim !== modelConfig.headDim;
  const hasSharedKvLayers = Number.isFinite(modelConfig.numKvSharedLayers)
    && modelConfig.numKvSharedLayers > 0;
  const hasExplicitLayerTypes = Array.isArray(modelConfig.layerTypes)
    && modelConfig.layerTypes.length === modelConfig.numLayers;
  return hasExplicitLayerTypes && (hasMixedAttentionGeometry || hasSharedKvLayers);
}

function getRequiredKernelMaxKVLen(operation, variant, label) {
  const config = getKernelConfig(operation, variant);
  const maxKVLen = config.variantMetadata?.maxKVLen;
  if (!Number.isFinite(maxKVLen) || maxKVLen <= 0) {
    throw new Error(`${label} kernel "${variant}" is missing variantMetadata.maxKVLen.`);
  }
  return maxKVLen;
}

function resolveTieredRequestedQuantMode(runtimeKV) {
  const tieringMode = String(runtimeKV?.tiering?.mode ?? 'off').trim().toLowerCase();
  assertSupportedTurboQuantMode(tieringMode, 'runtime.inference.session.kvcache.tiering.mode');
  const compressionMode = String(
    runtimeKV?.tiering?.compression?.mode
    ?? (tieringMode === 'int8'
      || tieringMode === 'int4'
      || tieringMode === 'turboquant'
      || tieringMode === 'turboquant_prod'
      ? tieringMode
      : 'none')
  ).trim().toLowerCase();
  assertSupportedTurboQuantMode(
    compressionMode,
    'runtime.inference.session.kvcache.tiering.compression.mode'
  );
  const gatingMode = String(runtimeKV?.tiering?.gating?.mode ?? 'auto').trim().toLowerCase();
  if (gatingMode === 'force_off') {
    return 'none';
  }
  return compressionMode;
}

function resolveContiguousRequestedQuantMode(runtimeKV) {
  const quantMode = String(runtimeKV?.quantization?.mode ?? 'none').trim().toLowerCase();
  assertSupportedTurboQuantMode(
    quantMode,
    'runtime.inference.session.kvcache.quantization.mode'
  );
  return quantMode;
}

function assertQuantizedKVKernelSupport(modelConfig, cacheLayout, cacheMaxSeqLen, runtimeKV) {
  if (cacheLayout === 'contiguous_quantized') {
    if (modelConfig.headDim > 256) {
      throw new Error(
        `Contiguous quantized KV cache requires headDim <= 256; got ${modelConfig.headDim}.`
      );
    }
    const quantMode = resolveContiguousRequestedQuantMode(runtimeKV);
    const variant = selectKernelRuleValue('attention', 'contiguousQuantVariant', { mode: quantMode });
    const maxKVLen = getRequiredKernelMaxKVLen(
      'attention_contiguous_quant',
      variant,
      'Contiguous quant attention'
    );
    if (cacheMaxSeqLen > maxKVLen) {
      throw new Error(
        `Contiguous quantized KV cache requires maxSeqLen <= ${maxKVLen}; got ${cacheMaxSeqLen}.`
      );
    }
    return;
  }

  if (cacheLayout !== 'tiered') {
    return;
  }

  const coldQuantMode = resolveTieredRequestedQuantMode(runtimeKV);
  if (coldQuantMode === 'none') {
    return;
  }
  if (modelConfig.headDim > 256) {
    throw new Error(
      `Tiered quantized KV cache requires headDim <= 256; got ${modelConfig.headDim}.`
    );
  }
  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode: coldQuantMode });
  const maxKVLen = getRequiredKernelMaxKVLen(
    'attention_tiered_quant',
    variant,
    'Tiered quant attention'
  );
  if (cacheMaxSeqLen > maxKVLen) {
    throw new Error(
      `Tiered quantized KV cache requires maxSeqLen <= ${maxKVLen}; got ${cacheMaxSeqLen}.`
    );
  }
}


// ============================================================================
// KV Cache Setup
// ============================================================================


export function createKVCache(modelConfig, useGPU, debug = false, runtimeConfig) {
  if (modelConfig?.decodeStrategy === 'replay_prefill') {
    throw new Error(
      'Live KV cache creation is not supported for models that require replay-prefill decode. ' +
      'Skip createKVCache() when the model config does not resolve explicit layerTypes for mixed-geometry/shared-KV decode.'
    );
  }
  const runtimeKV = resolveRuntimeKVConfig(runtimeConfig);
  const requiresMixedGeometryKVCache = usesMixedGeometryKVCache(modelConfig);
  const contiguousKVPolicy = resolveContiguousKVPolicy(modelConfig);
  const forceContiguousKVCache = contiguousKVPolicy.forceContiguousKVCache;
  const modelMaxSeqLen = modelConfig.maxSeqLen;
  if (!Number.isFinite(modelMaxSeqLen) || modelMaxSeqLen <= 0) {
    throw new Error('Model config is missing maxSeqLen.');
  }
  let slidingWindow = modelConfig.slidingWindow;

  let cacheMaxSeqLen = modelMaxSeqLen;
  if (Number.isFinite(runtimeKV.maxSeqLen) && runtimeKV.maxSeqLen > 0) {
    cacheMaxSeqLen = Math.min(cacheMaxSeqLen, runtimeKV.maxSeqLen);
  }

  
  let cacheLayout = runtimeKV.layout;
  if (!cacheLayout) {
    throw new Error('runtime.inference.session.kvcache.layout is required.');
  }
  if (cacheLayout === 'tiered' && !runtimeKV.tiering) {
    throw new Error('runtime.inference.session.kvcache.tiering is required for tiered layout.');
  }
  const tieringMode = runtimeKV.tiering?.mode;
  if (tieringMode == null) {
    throw new Error('runtime.inference.session.kvcache.tiering.mode is required.');
  }
  let layoutSource = 'runtime';
  if (tieringMode !== 'off' && cacheLayout !== 'tiered') {
    if (cacheLayout !== 'contiguous') {
      throw new Error('runtime.inference.session.kvcache.layout must be "tiered" when tiering.mode is enabled.');
    }
    cacheLayout = 'tiered';
    layoutSource = 'tiering';
  }
  if (!forceContiguousKVCache && cacheLayout === 'contiguous' && cacheMaxSeqLen >= PAGED_LAYOUT_SEQ_LEN_THRESHOLD) {
    cacheLayout = 'paged';
    layoutSource = 'threshold';
  }
  const quantMode = resolveContiguousRequestedQuantMode(runtimeKV);
  if (forceContiguousKVCache && cacheLayout === 'contiguous' && quantMode !== 'none') {
    cacheLayout = 'contiguous_quantized';
    layoutSource = 'quantization';
  }
  if (forceContiguousKVCache && cacheLayout === 'paged') {
    throw new Error(
      'Paged KV cache layout is not supported for models with full-attention layers. ' +
      'Set runtime.inference.session.kvcache.layout to "contiguous" instead.'
    );
  }
  if (requiresMixedGeometryKVCache) {
    if (cacheLayout !== 'contiguous') {
      throw new Error(
        `Mixed-geometry incremental KV cache requires layout="contiguous"; got "${cacheLayout}". ` +
        'Disable tiering and quantized KV cache for this model.'
      );
    }
    if (runtimeKV.tiering?.mode != null && runtimeKV.tiering.mode !== 'off') {
      throw new Error(
        `Mixed-geometry incremental KV cache requires tiering.mode="off"; got "${runtimeKV.tiering.mode}".`
      );
    }
    if (resolveContiguousRequestedQuantMode(runtimeKV) !== 'none') {
      throw new Error(
        'Mixed-geometry incremental KV cache does not support contiguous quantization yet. ' +
        'Set runtime.inference.session.kvcache.quantization.mode="none".'
      );
    }
    if (!useGPU) {
      throw new Error(
        'Mixed-geometry incremental KV cache requires GPU execution. ' +
        'Use a WebGPU-capable surface.'
      );
    }
  }
  if (debug && cacheLayout !== runtimeKV.layout) {
    log.debug('Pipeline', `KV cache layout override: ${runtimeKV.layout} -> ${cacheLayout} (${layoutSource})`);
  }

  // Sliding-window attention only needs a bounded KV cache on contiguous layouts.
  if (slidingWindow && Number.isFinite(slidingWindow) && slidingWindow > 0) {
    if (runtimeKV.windowSize > 0) {
      slidingWindow = Math.min(slidingWindow, runtimeKV.windowSize);
    }
    if (!forceContiguousKVCache && cacheLayout !== 'paged' && cacheLayout !== 'tiered') {
      cacheMaxSeqLen = Math.min(cacheMaxSeqLen, slidingWindow);
    }
  }

  // Use f16 KV cache when supported to reduce VRAM.
  // For models with attention logit softcapping, allow forcing F32 via runtime config
  // to avoid precision issues in attention. See: https://github.com/ggerganov/llama.cpp/issues/8853
  const gpuCaps = getKernelCapabilities();
  // Use config value directly instead of model detection flag (manifest-first architecture)
  // Check > 0 to allow explicit "disabled" encoding as 0 or null
  const attnSoftcap = modelConfig.attnLogitSoftcapping;
  const hasAttnSoftcapping = attnSoftcap != null && attnSoftcap > 0;
  const forceF32Softcap = runtimeKV.forceF32Softcap === true;
  const forceF32KV = hasAttnSoftcapping && forceF32Softcap;
  
  const kvDtype = selectRuleValue('inference', 'dtype', 'kvCacheDtype', {
    requested: runtimeKV.kvDtype,
    useGPU,
    hasF16: gpuCaps.hasF16,
    forceF32: forceF32KV,
  });
  if (forceF32KV && debug) {
    log.debug('Pipeline', `Forcing F32 KV cache (attnLogitSoftcapping=${modelConfig.attnLogitSoftcapping}, forceF32Softcap=true)`);
  }
  if (cacheLayout === 'tiered' && kvDtype !== 'f16') {
    throw new Error('Tiered KV cache requires kvDtype="f16" (no f32 tiered kernels yet).');
  }
  if (cacheLayout === 'contiguous_quantized' && kvDtype !== 'f16') {
    throw new Error('Contiguous quantized KV cache requires kvDtype="f16".');
  }
  if (cacheLayout === 'contiguous_quantized' && !useGPU) {
    throw new Error('Contiguous quantized KV cache requires GPU.');
  }

  if (useGPU && (cacheLayout === 'paged' || cacheLayout === 'tiered' || cacheLayout === 'bdpa' || cacheLayout === 'contiguous_quantized')) {
    const limits = getDeviceLimits();
    if (limits) {
      const maxLayerHeadDim = Math.max(
        modelConfig.headDim,
        Number.isFinite(modelConfig.globalHeadDim) && modelConfig.globalHeadDim > 0
          ? modelConfig.globalHeadDim
          : modelConfig.headDim
      );
      const bytesPerToken = modelConfig.numKVHeads * maxLayerHeadDim * (kvDtype === 'f16' ? 2 : 4);
      const maxByBinding = Math.floor(limits.maxStorageBufferBindingSize / bytesPerToken);
      const maxByBuffer = Math.floor(limits.maxBufferSize / bytesPerToken);
      const fallbackMax = Number.isFinite(runtimeKV.gpuPagedFallbackMaxSeqLen) && runtimeKV.gpuPagedFallbackMaxSeqLen > 0
        ? runtimeKV.gpuPagedFallbackMaxSeqLen
        : Infinity;
      const limitMax = Math.min(maxByBinding, maxByBuffer, fallbackMax);
      if (!Number.isFinite(limitMax) || limitMax <= 0) {
        throw new Error('KV cache maxSeqLen exceeds device buffer limits.');
      }
      if (Number.isFinite(limitMax) && limitMax > 0 && limitMax < cacheMaxSeqLen) {
        log.warn(
          'Pipeline',
          `KV cache maxSeqLen capped ${cacheMaxSeqLen} -> ${limitMax} (layout=${cacheLayout}, limit=${limits.maxStorageBufferBindingSize}).`
        );
        cacheMaxSeqLen = limitMax;
      }
    }
  }

  assertQuantizedKVKernelSupport(modelConfig, cacheLayout, cacheMaxSeqLen, runtimeKV);

  
	  const cacheConfig = {
	    numLayers: modelConfig.numLayers,
	    numHeads: modelConfig.numKVHeads,
	    headDim: modelConfig.headDim,
	    maxSeqLen: cacheMaxSeqLen,
	    useGPU,
	    layout: cacheLayout,
	    kvDtype,
	    bdpaVocabSize: runtimeKV.bdpaVocabSize,
	    pageSize: runtimeKV.pageSize,
	  };

  
  let kvCache;

  if (requiresMixedGeometryKVCache) {
    kvCache = new MixedGeometryKVCache({
      ...cacheConfig,
      numHeads: modelConfig.numKVHeads,
      globalHeadDim: modelConfig.globalHeadDim ?? null,
      slidingWindow,
      layerTypes: modelConfig.layerTypes,
    });
  } else if (modelConfig.slidingWindow && !forceContiguousKVCache && cacheLayout !== 'paged' && cacheLayout !== 'tiered' && cacheLayout !== 'bdpa') {
    kvCache = new SlidingWindowKVCache({
      ...cacheConfig,
      windowSize: slidingWindow ?? modelConfig.slidingWindow,
    });
  } else if (cacheLayout === 'bdpa') {
    kvCache = new BasisDecomposedPagedCache({
      ...cacheConfig,
    });
  } else if (cacheLayout === 'tiered') {
    kvCache = new TieredKVCache({
      ...cacheConfig,
      tiering: runtimeKV.tiering,
    });
  } else if (cacheLayout === 'contiguous_quantized') {
    const quantCfg = runtimeKV.quantization ?? {};
    const qCache = new QuantizedKVCache({
      ...cacheConfig,
      quantMode,
      bitWidth: quantCfg.bitWidth ?? 4,
      prodMode: quantCfg.prodMode === true,
    });
    const device = getDevice();
    qCache.setSharedBuffers(retainTurboQuantSharedBuffers(device, {
      headDim: modelConfig.headDim,
      bitWidth: quantCfg.bitWidth ?? 4,
      prodMode: quantCfg.prodMode === true,
    }));
    kvCache = qCache;
  } else {
    kvCache = new KVCache(cacheConfig);
  }

  // Diagnostic logging: actual vs requested KV cache parameters
  const requestedKvDtype = runtimeKV.kvDtype ?? 'unset';
  const requestedMaxSeqLen = runtimeKV.maxSeqLen ?? 'unset';
  const actualKvDtype = kvDtype;
  const actualMaxSeqLen = cacheMaxSeqLen;

  if (requestedKvDtype !== 'unset' && requestedKvDtype !== actualKvDtype) {
    log.warn(
      'Pipeline',
      `KV cache kvDtype mismatch: requested=${requestedKvDtype}, actual=${actualKvDtype} ` +
      `(rule-based resolution may have changed the dtype due to GPU capabilities or softcap policy)`
    );
  }
  if (requestedMaxSeqLen !== 'unset' && requestedMaxSeqLen !== actualMaxSeqLen) {
    log.debug(
      'Pipeline',
      `KV cache maxSeqLen adjusted: requested=${requestedMaxSeqLen}, actual=${actualMaxSeqLen}`
    );
  }

  log.info(
    'Pipeline',
    `KV cache allocated: kvDtype=${actualKvDtype}, maxSeqLen=${actualMaxSeqLen}, ` +
    `layout=${cacheLayout}, modelMaxSeqLen=${modelMaxSeqLen}, ` +
    `requestedKvDtype=${requestedKvDtype}, requestedMaxSeqLen=${requestedMaxSeqLen}`
  );

  if (debug) {
    if (contiguousKVPolicy.reason === 'layerPattern') {
      log.debug('Pipeline', 'Layer pattern includes full-attention layers; paged layout blocked, contiguous enforced.');
    } else if (contiguousKVPolicy.reason === 'slidingWindow') {
      log.debug('Pipeline', 'Model declares attention.slidingWindow=null without explicit layerTypes; treating KV layout as contiguous-compatible.');
    }
    const isSliding = kvCache instanceof SlidingWindowKVCache;
    log.debug('Pipeline', `KV cache: type=${kvCache?.constructor?.name || 'unknown'}, kvDtype=${kvCache.kvDtype}, layout=${kvCache.layout}, maxSeqLen=${kvCache.maxSeqLen}, windowSize=${isSliding ? kvCache.windowSize : null}`);
  }

  return kvCache;
}

// ============================================================================
// Tokenizer Setup
// ============================================================================


export async function initTokenizer(manifest, options = {}) {
  const { baseUrl, tokenizerHints, storageContext } = options;
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(manifest, {
    baseUrl,
    tokenizerHints,
    loadTokenizerJson: typeof storageContext?.loadTokenizerJson === 'function'
      ? () => storageContext.loadTokenizerJson()
      : null,
    loadTokenizerModel: typeof storageContext?.loadTokenizerModel === 'function'
      ? (path) => storageContext.loadTokenizerModel(path)
      : null,
  });
  return tokenizer;
}

// ============================================================================
// Weight Loading
// ============================================================================


export async function loadWeights(manifest, modelConfig, options = {}) {
  const {
    onProgress,
    loadingConfig,
    baseUrl,
    loaderDebug,
    perLayerInputSession,
  } = options;
  const runtimeStorageContext = options.storageContext
    ?? createRemoteStorageContext(baseUrl, manifest);
  const verifyHashes = (
    typeof runtimeStorageContext?.verifyHashes === 'boolean'
      ? runtimeStorageContext.verifyHashes
      : options.verifyHashes
  ) ?? loadingConfig?.shardCache?.verifyHashes;
  if (verifyHashes == null) {
    throw new Error('runtime.loading.shardCache.verifyHashes is required.');
  }

  const dopplerLoader = getDopplerLoader(loadingConfig);
  const keepF32Weights = options.keepF32Weights === true;
  dopplerLoader.setQ4KConfig(
    resolveQ4KConfig(
      manifest,
      options.resolvedKernelPath ?? null,
      options.kernelPathSource ?? 'none',
      keepF32Weights
    )
  );
  dopplerLoader.setLoaderDebugConfig(loaderDebug ?? null);
  dopplerLoader.setPerLayerInputSession(
    resolvePerLayerInputsSession(
      modelConfig.perLayerInputsSession ?? null,
      perLayerInputSession ?? null
    )
  );

  const tensorsFile = isRDRRManifest(manifest) ? manifest.tensorsFile : null;
  if (baseUrl && tensorsFile) {
    const base = baseUrl.replace(/\/$/, '');
    const filename = tensorsFile.replace(/^\/+/, '');
    dopplerLoader.setTensorsJsonUrl(`${base}/${filename}`);
  } else {
    dopplerLoader.setTensorsJsonUrl(null);
  }

  // Configure custom shard loader if provided (Native Bridge or direct-source bundle)
  const hasLoadShard = typeof runtimeStorageContext?.loadShard === 'function';
  const hasLoadShardRange = typeof runtimeStorageContext?.loadShardRange === 'function';
  const hasStreamShardRange = typeof runtimeStorageContext?.streamShardRange === 'function';
  if (hasLoadShard || hasLoadShardRange) {
    log.debug('Pipeline', 'Using custom shard loader (Native Bridge or external)');

    const loadShard = async (index) => {
      if (hasLoadShard) {
        const data = await runtimeStorageContext.loadShard(index);
        return toUint8Array(data, 'storageContext.loadShard');
      }
      const rangeData = await runtimeStorageContext.loadShardRange(index, 0, null);
      return toUint8Array(rangeData, 'storageContext.loadShardRange');
    };

    const loadShardRange = hasLoadShardRange
      ? async (index, offset, length = null) => {
        const data = await runtimeStorageContext.loadShardRange(index, offset, length);
        return toArrayBuffer(data, 'storageContext.loadShardRange');
      }
      : null;

    const streamShardRange = hasStreamShardRange
      ? async function* (index, offset = 0, length = null, streamOptions = {}) {
        for await (const chunk of runtimeStorageContext.streamShardRange(index, offset, length, streamOptions)) {
          yield toUint8Array(chunk, 'storageContext.streamShardRange');
        }
      }
      : null;

    dopplerLoader.setCustomShardLoader(loadShard, {
      verify: verifyHashes,
      loadShardRange,
      streamShardRange,
    });
    if (isRDRRManifest(manifest)) {
      dopplerLoader.setManifest(manifest);
    }
  }

  await dopplerLoader.init();

  // Load model via DopplerLoader
  const modelId = manifest.modelId;
  if (!modelId) {
    throw new Error('Manifest is missing modelId. Re-convert the model with modelId set.');
  }
  await dopplerLoader.load(modelId, {
    verifyHashes,
    onProgress: onProgress || ((info) => {
      // Shard and layer progress are logged by loader with source info
      if (info.stage !== 'layers' && info.stage !== 'shards') {
        log.verbose('Loader', `${info.stage}: ${Math.round(info.progress * 100)}%`);
      }
    }),
  });

  // Map layer weights
  
  const layerWeights = new Map();
  for (let l = 0; l < modelConfig.numLayers; l++) {
    const weights = dopplerLoader.getLayerWeights(l);
    if (weights) {
      layerWeights.set(`layer_${l}`, weights);
    }
  }

  // Collect per-layer router weights for MoE
  
  const layerRouterWeights = new Map();
  if (modelConfig.useMoE) {
    for (let l = 0; l < modelConfig.numLayers; l++) {
      const weights = layerWeights.get(`layer_${l}`);
      if (weights?.routerWeight) {
        layerRouterWeights.set(l, {
          weight: weights.routerWeight,
          bias: weights.routerBias || null,
        });
      }
    }
    log.debug('Pipeline', 'MoE model - experts will be loaded on demand');
  }

  return {
    layerWeights,
    embeddings: dopplerLoader.embeddings,
    lmHead: dopplerLoader.lmHead,
    finalNorm: dopplerLoader.finalNorm,
    embeddingPostprocessor: dopplerLoader.embeddingPostprocessor,
    perLayerInputWeights: dopplerLoader.perLayerInputWeights,
    layerRouterWeights,
  };
}

// ============================================================================
// MoE Router Setup
// ============================================================================


export function initMoERouter(modelConfig, moeRoutingConfig, layerWeights) {
  if (!modelConfig.useMoE) return null;

  const router = new MoERouter({
    numExperts: modelConfig.numExperts,
    topK: modelConfig.moeTopK,
    hiddenSize: modelConfig.hiddenSize,
    normalizeWeights: moeRoutingConfig.normalizeWeights,
  });

  // Find first layer with router weights
  for (let l = 0; l < modelConfig.numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (weights?.routerWeight) {
      router.loadWeights(weights.routerWeight, weights.routerBias || null);
      log.debug('Pipeline', `Loaded MoE router from layer ${l}${weights.routerBias ? ' (with bias)' : ''}`);
      break;
    }
  }

  return router;
}

// ============================================================================
// Speculative Decoder Setup
// ============================================================================


// EXPERIMENTAL: Speculative decoding is parsed and initialized but the full
// verify-and-accept loop is not yet wired into the generation pipeline.
// Enabling this will create the decoder state but decoded tokens are not
// verified against the draft model. Do not rely on this for production use.
export function initSpeculativeDecoder(manifest, speculativeConfig) {
  if (!manifest.draftModel) return null;
  if (manifest.draftModel.numTokens == null) {
    throw new Error(`Manifest "${manifest.modelId}" is missing draftModel.numTokens.`);
  }

  log.warn(
    'Pipeline',
    `Speculative decoding enabled for "${manifest.modelId}" but this feature is experimental and not fully wired. ` +
    'The draft-verify-accept loop is incomplete. Generated output is from the base model only.'
  );

  return new SpeculativeDecoder({
    numDraftTokens: manifest.draftModel.numTokens,
    maxRejectionRetries: speculativeConfig.maxRejectionRetries,
    enableTreeDraft: speculativeConfig.enableTreeDraft,
    temperature: speculativeConfig.temperature,
    randomSeed: speculativeConfig.randomSeed,
  });
}
// ============================================================================
// Emulation Setup
// ============================================================================

export async function initEmulation(runtimeConfig) {
  const emulationConfig = runtimeConfig?.emulation;

  // Skip if emulation is not enabled
  if (!emulationConfig?.enabled) {
    return null;
  }

  try {
    const simulatorModuleRoot = '/proto/simulator';
    const simulatorEnvSpecifier = `${simulatorModuleRoot}/env.js`;
    const simulatorIndexSpecifier = `${simulatorModuleRoot}/index.js`;

    // Dynamically import to avoid loading emulation code when disabled
    const { setSimulatorEnv } = await import(simulatorEnvSpecifier);
    const { createEmulationConfig, formatBytes, formatBandwidth } = await import('../../../config/schema/emulation.schema.js');
    const { EmulatedVramStore, detectLocalResources } = await import('../../../storage/emulated-vram.js');
    const { getBufferPool } = await import('../../../memory/buffer-pool.js');
    const { createEmulationContext, isEmulationSupported } = await import(simulatorIndexSpecifier);

    setSimulatorEnv({
      log,
      bufferPool: getBufferPool,
      createEmulationConfig,
      formatBytes,
      formatBandwidth,
      detectLocalResources,
      createVramStore: (config, budgets) =>
        new EmulatedVramStore(config.opfsRootPath, budgets.vramBudgetBytes, budgets.ramBudgetBytes),
    });

    const supported = await isEmulationSupported();
    if (!supported) {
      throw new Error('Emulation requested but not supported in this environment.');
    }

    // Create emulation context
    log.info('Pipeline', `Initializing emulation for ${emulationConfig.targetChip}`);
    const ctx = await createEmulationContext(emulationConfig);

    log.info('Pipeline', `Emulation ready: ${ctx.config.topology.gpuCount} virtual GPUs, timing mode: ${ctx.config.timingMode}`);

    return ctx;
  } catch (err) {
    const message = resolveErrorMessage(err);
    log.error('Pipeline', `Failed to initialize emulation: ${message}`);
    throw new Error(`Failed to initialize emulation: ${message}`);
  }
}

export async function destroyEmulation(emulation) {
  if (emulation) {
    try {
      await emulation.destroy();
      log.info('Pipeline', 'Emulation context destroyed');
    } catch (err) {
      log.warn('Pipeline', `Error destroying emulation: ${err.message}`);
    }
  }
}
