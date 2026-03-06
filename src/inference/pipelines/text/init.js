

import { parseModelConfig } from './config.js';
import { getDevice, getDeviceLimits, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { KVCache, SlidingWindowKVCache, TieredKVCache, BasisDecomposedPagedCache } from '../../kv-cache.js';
import { Tokenizer } from '../../tokenizer.js';
import { MoERouter } from '../../moe-router.js';
import { SpeculativeDecoder } from '../../speculative.js';
import { getDopplerLoader } from '../../../loader/doppler-loader.js';
import { log, setGPUDevice, trace as debugTrace } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import { PAGED_LAYOUT_SEQ_LEN_THRESHOLD } from '../../../config/schema/index.js';
import { isKernelPathFusedQ4K } from '../../../config/kernel-path-loader.js';
import { createWeightBuffer, getWeightDtype, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

function resolveErrorMessage(error) {
  if (error && typeof error === 'object' && typeof error.message === 'string') {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return String(error);
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

function normalizeBaseUrl(baseUrl) {
  if (typeof baseUrl !== 'string' || baseUrl.trim().length === 0) {
    return null;
  }
  return baseUrl.replace(/\/$/, '');
}

function createRemoteStorageContext(baseUrl, manifest) {
  const root = normalizeBaseUrl(baseUrl);
  if (!root || !isRDRRManifest(manifest)) {
    return null;
  }

  return {
    async loadShard(index) {
      const shard = manifest.shards[index];
      const filename = shard?.filename;
      if (!filename) {
        throw new Error(`Manifest shard ${index} is missing filename.`);
      }
      const response = await fetch(`${root}/${filename.replace(/^\/+/, '')}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch shard ${index} from ${root}: ${response.status}`);
      }
      return new Uint8Array(await response.arrayBuffer());
    },
  };
}


function resolveQ4KConfig(
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
  if (q4kLayout != null && q4kLayout !== 'row' && q4kLayout !== 'col') {
    throw new Error(
      `Manifest "${manifest?.modelId ?? 'unknown'}" has invalid quantizationInfo.layout "${q4kLayout}". Expected "row" or "col".`
    );
  }
  let useFused = kernelPath ? isKernelPathFusedQ4K(kernelPath) : hasSubgroups;
  if (q4kLayout === 'col') {
    useFused = false;
  }

  const pathLabel = kernelPath?.id ?? 'auto';
  const layoutLabel = q4kLayout ?? 'none';
  debugTrace.loader(`Q4K config: fused=${useFused}, kernelPath=${pathLabel}, source=${kernelPathSource}, layout=${layoutLabel}, subgroups=${hasSubgroups}`);

  return {
    useFusedQ4K: useFused,
    q4kLayout,
    keepF32Weights,
  };
}

// ============================================================================
// RoPE Initialization
// ============================================================================


function computeRoPEFreqsForTheta(theta, headDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling) {
  const halfDim = headDim / 2;

  // Compute base frequencies: theta_i = 1 / (base^(2i/d))
  const freqs = new Float32Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    freqs[i] = 1.0 / Math.pow(theta, (2 * i) / headDim);
  }

  // Compute per-dimension scaling factors
  const scales = new Float32Array(halfDim);
  const isYarn = ropeScalingType === 'yarn';
  if (isYarn) {
    // YARN scaling - validate ALL required params (fail fast on incomplete manifest)
    if (ropeScaling?.factor == null ||
        ropeScaling?.beta_fast == null || ropeScaling?.beta_slow == null ||
        ropeScaling?.original_max_position_embeddings == null) {
      throw new Error(
        `RoPE scaling type is 'yarn' but YARN params missing. ` +
        `Manifest must provide factor, beta_fast, beta_slow, and original_max_position_embeddings. ` +
        `Got: factor=${ropeScaling?.factor}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}, ` +
        `original_max_position_embeddings=${ropeScaling?.original_max_position_embeddings}`
      );
    }
    // Extract validated YARN params (no hidden defaults - all guaranteed non-null)
    const yarnFactor = ropeScaling.factor;
    const yarnBetaFast = ropeScaling.beta_fast;
    const yarnBetaSlow = ropeScaling.beta_slow;
    const originalMaxPos = ropeScaling.original_max_position_embeddings;

    // YARN: wavelength-based interpolation
    for (let i = 0; i < halfDim; i++) {
      const wavelength = (2 * Math.PI) / freqs[i];
      const lowThresh = originalMaxPos / yarnBetaSlow;
      const highThresh = originalMaxPos / yarnBetaFast;

      if (wavelength < highThresh) {
        scales[i] = 1.0;
      } else if (wavelength > lowThresh) {
        scales[i] = yarnFactor;
      } else {
        const t = (wavelength - highThresh) / (lowThresh - highThresh);
        scales[i] = 1.0 + (yarnFactor - 1.0) * t;
      }
    }
  } else {
    // Linear scaling: uniform across all dimensions
    for (let i = 0; i < halfDim; i++) {
      scales[i] = ropeScale;
    }
  }

  // Compute cos/sin for each position
  const cosValues = new Float32Array(maxSeqLen * halfDim);
  const sinValues = new Float32Array(maxSeqLen * halfDim);

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const scaledPos = pos / scales[i];
      const angle = scaledPos * freqs[i];
      cosValues[pos * halfDim + i] = Math.cos(angle);
      sinValues[pos * halfDim + i] = Math.sin(angle);
    }
  }

  return { cos: cosValues, sin: sinValues };
}

function isSameRoPEScalingConfig(
  leftType,
  leftScale,
  leftScaling,
  rightType,
  rightScale,
  rightScaling
) {
  if (leftType !== rightType) return false;
  if (leftScale !== rightScale) return false;
  if (leftType !== 'yarn') return true;
  return (leftScaling?.beta_fast ?? null) === (rightScaling?.beta_fast ?? null)
    && (leftScaling?.beta_slow ?? null) === (rightScaling?.beta_slow ?? null)
    && (leftScaling?.original_max_position_embeddings ?? null)
      === (rightScaling?.original_max_position_embeddings ?? null);
}

function resolveRotaryDim(headDim, rotaryDim, partialRotaryFactor) {
  if (rotaryDim != null) {
    if (!Number.isFinite(rotaryDim) || rotaryDim <= 0 || (rotaryDim % 2) !== 0) {
      throw new Error(`RoPE rotary dim must be a positive even integer; got "${rotaryDim}".`);
    }
    if (rotaryDim > headDim) {
      throw new Error(`RoPE rotary dim ${rotaryDim} cannot exceed headDim ${headDim}.`);
    }
    return rotaryDim;
  }
  if (partialRotaryFactor == null) {
    return headDim;
  }
  if (!Number.isFinite(partialRotaryFactor) || partialRotaryFactor <= 0 || partialRotaryFactor > 1) {
    throw new Error(
      `RoPE partialRotaryFactor must be a number in (0, 1]; got "${partialRotaryFactor}".`
    );
  }
  const resolved = Math.trunc(headDim * partialRotaryFactor);
  if (resolved <= 0 || (resolved % 2) !== 0) {
    throw new Error(
      `RoPE partialRotaryFactor=${partialRotaryFactor} with headDim=${headDim} resolves ` +
      `to rotaryDim=${resolved}, but rotaryDim must be a positive even integer.`
    );
  }
  return resolved;
}


export async function initRoPEFrequencies(config, useGPU) {
  const {
    headDim,
    rotaryDim,
    maxSeqLen,
    ropeTheta,
    ropeLocalTheta,
    mropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeScale,
    ropeLocalScale,
    ropeScalingType,
    ropeLocalScalingType,
    ropeScaling,
    ropeLocalScaling,
  } = config;
  if (!Number.isFinite(ropeScale) || ropeScale <= 0) {
    throw new Error(`RoPE scale must be a positive number; got "${ropeScale}".`);
  }
  const resolvedLocalScale = ropeLocalScale ?? ropeScale;
  if (!Number.isFinite(resolvedLocalScale) || resolvedLocalScale <= 0) {
    throw new Error(`Local RoPE scale must be a positive number; got "${resolvedLocalScale}".`);
  }
  const resolvedLocalTheta = ropeLocalTheta ?? ropeTheta;
  const resolvedLocalScalingType = ropeLocalScalingType ?? ropeScalingType;
  const resolvedLocalScaling = ropeLocalScaling ?? ropeScaling;
  const resolvedRotaryDim = resolveRotaryDim(headDim, rotaryDim, partialRotaryFactor);
  const halfDim = resolvedRotaryDim / 2;
  if (mropeInterleaved === true && Array.isArray(mropeSection)) {
    const expandedDim = mropeSection.reduce((sum, entry) => sum + entry, 0) * 2;
    if (expandedDim !== resolvedRotaryDim) {
      throw new Error(
        `RoPE mropeSection expands to ${expandedDim} dims, but rotaryDim is ${resolvedRotaryDim}.`
      );
    }
  }

  const isYarn = ropeScalingType === 'yarn';
  const isLocalYarn = resolvedLocalScalingType === 'yarn';

  // Compute global (full_attention) frequencies
  const globalFreqs = computeRoPEFreqsForTheta(
    ropeTheta, resolvedRotaryDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling
  );

  // Compute local (sliding_attention) frequencies if different from global.
  // Models with dual RoPE use different theta for local vs global attention layers.
  
  let localFreqs = null;
  const hasDistinctLocalTheta = resolvedLocalTheta !== ropeTheta;
  const hasDistinctLocalScaling = !isSameRoPEScalingConfig(
    ropeScalingType,
    ropeScale,
    ropeScaling,
    resolvedLocalScalingType,
    resolvedLocalScale,
    resolvedLocalScaling
  );
  if (hasDistinctLocalTheta || hasDistinctLocalScaling) {
    localFreqs = computeRoPEFreqsForTheta(
      resolvedLocalTheta,
      resolvedRotaryDim,
      maxSeqLen,
      resolvedLocalScale,
      resolvedLocalScalingType,
      resolvedLocalScaling
    );
    log.debug(
      'Pipeline',
      `Dual RoPE: local theta=${resolvedLocalTheta}, global theta=${ropeTheta}, ` +
      `localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}, ` +
      `globalScaling=${ropeScalingType ?? 'none'}:${ropeScale}`
    );
  }

  if (isYarn) {
    // Log YARN params (already validated in computeRoPEFreqs)
    log.debug('Pipeline', `YARN RoPE: factor=${ropeScaling?.factor}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}`);
  }
  if (isLocalYarn && hasDistinctLocalScaling) {
    log.debug(
      'Pipeline',
      `Local YARN RoPE: factor=${resolvedLocalScaling?.factor}, ` +
      `beta_fast=${resolvedLocalScaling?.beta_fast}, beta_slow=${resolvedLocalScaling?.beta_slow}`
    );
  }

  // Upload to GPU if available
  const device = getDevice();
  if (device && useGPU) {
    const cosBuffer = acquireBuffer(globalFreqs.cos.byteLength, undefined, 'rope_cos');
    const sinBuffer = acquireBuffer(globalFreqs.sin.byteLength, undefined, 'rope_sin');
    device.queue.writeBuffer(cosBuffer, 0, globalFreqs.cos.buffer, globalFreqs.cos.byteOffset, globalFreqs.cos.byteLength);
    device.queue.writeBuffer(sinBuffer, 0, globalFreqs.sin.buffer, globalFreqs.sin.byteOffset, globalFreqs.sin.byteLength);

    
    let localCosBuffer;
    
    let localSinBuffer;
    if (localFreqs) {
      localCosBuffer = acquireBuffer(localFreqs.cos.byteLength, undefined, 'rope_local_cos');
      localSinBuffer = acquireBuffer(localFreqs.sin.byteLength, undefined, 'rope_local_sin');
      device.queue.writeBuffer(localCosBuffer, 0, localFreqs.cos.buffer, localFreqs.cos.byteOffset, localFreqs.cos.byteLength);
      device.queue.writeBuffer(localSinBuffer, 0, localFreqs.sin.buffer, localFreqs.sin.byteOffset, localFreqs.sin.byteLength);
    }

    log.debug(
      'Pipeline',
      `RoPE frequencies initialized (GPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, rotaryDim=${resolvedRotaryDim}, ` +
      `theta=${ropeTheta}${hasDistinctLocalTheta ? `, localTheta=${resolvedLocalTheta}` : ''}, ` +
      `scaling=${ropeScalingType ?? 'none'}:${ropeScale}${hasDistinctLocalScaling ? `, localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}` : ''}, ` +
      `interleaved=${mropeInterleaved === true}`
    );

    return {
      cos: cosBuffer,
      sin: sinBuffer,
      localCos: localCosBuffer,
      localSin: localSinBuffer,
    };
  }

  log.debug(
    'Pipeline',
    `RoPE frequencies initialized (CPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, rotaryDim=${resolvedRotaryDim}, ` +
    `theta=${ropeTheta}${hasDistinctLocalTheta ? `, localTheta=${resolvedLocalTheta}` : ''}, ` +
    `scaling=${ropeScalingType ?? 'none'}:${ropeScale}${hasDistinctLocalScaling ? `, localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}` : ''}, ` +
    `interleaved=${mropeInterleaved === true}`
  );

  return {
    cos: globalFreqs.cos,
    sin: globalFreqs.sin,
    localCos: localFreqs?.cos,
    localSin: localFreqs?.sin,
  };
}


export function isGPURoPEBuffers(buffers) {
  if (typeof GPUBuffer === 'undefined') return false;
  return !!buffers?.cos && buffers.cos instanceof GPUBuffer;
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


// ============================================================================
// KV Cache Setup
// ============================================================================


export function createKVCache(modelConfig, useGPU, debug = false, runtimeConfig) {
  const runtimeKV = runtimeConfig ?? getRuntimeConfig().inference.kvcache;
  const forceContiguousKVCache = hasFullAttentionLayers(modelConfig.layerTypes);
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
    throw new Error('runtime.inference.kvcache.layout is required.');
  }
  if (cacheLayout === 'tiered' && !runtimeKV.tiering) {
    throw new Error('runtime.inference.kvcache.tiering is required for tiered layout.');
  }
  const tieringMode = runtimeKV.tiering?.mode;
  if (tieringMode == null) {
    throw new Error('runtime.inference.kvcache.tiering.mode is required.');
  }
  let layoutSource = 'runtime';
  if (tieringMode !== 'off' && cacheLayout !== 'tiered') {
    if (cacheLayout !== 'contiguous') {
      throw new Error('runtime.inference.kvcache.layout must be "tiered" when tiering.mode is enabled.');
    }
    cacheLayout = 'tiered';
    layoutSource = 'tiering';
  }
  if (!forceContiguousKVCache && cacheLayout === 'contiguous' && cacheMaxSeqLen >= PAGED_LAYOUT_SEQ_LEN_THRESHOLD) {
    cacheLayout = 'paged';
    layoutSource = 'threshold';
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

  if (useGPU && (cacheLayout === 'paged' || cacheLayout === 'tiered' || cacheLayout === 'bdpa')) {
    const limits = getDeviceLimits();
    if (limits) {
      const bytesPerToken = modelConfig.numKVHeads * modelConfig.headDim * (kvDtype === 'f16' ? 2 : 4);
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

  if (modelConfig.slidingWindow && !forceContiguousKVCache && cacheLayout !== 'paged' && cacheLayout !== 'tiered' && cacheLayout !== 'bdpa') {
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
  } else {
    kvCache = new KVCache(cacheConfig);
  }

  if (debug) {
    if (forceContiguousKVCache && modelConfig.layerTypes) {
      log.debug('Pipeline', 'Layer pattern includes full-attention layers; forcing contiguous KV cache.');
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
  const { baseUrl, presetTokenizer, storageContext } = options;
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(manifest, {
    baseUrl,
    presetTokenizer,
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
  const { onProgress, loadingConfig, baseUrl } = options;
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
    layerRouterWeights,
  };
}

// ============================================================================
// Chat Templates
// ============================================================================

// Simple prompt templates for single-turn chat.
// For multi-turn conversations, use formatChatMessages from chat-format.js.

function applyTurnBasedTemplate(prompt) {
  // Turn-based format: <start_of_turn>role\ncontent<end_of_turn>
  const userTurn = `<start_of_turn>user\n${prompt}<end_of_turn>\n`;
  const modelTurn = `<start_of_turn>model\n`;
  return userTurn + modelTurn;
}

function applyHeaderBasedTemplate(prompt) {
  // Header-based format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
  return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n${prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}

function applyChannelBasedTemplate(prompt) {
  // Channel-based format: <|start|>role<|message|>content<|end|>
  return `<|start|>user<|message|>${prompt}<|end|><|start|>assistant<|channel|>final<|message|>`;
}

function applyChatMLTemplate(prompt) {
  // ChatML format: <|im_start|>role\ncontent<|im_end|>
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}

function applyQwenTemplate(prompt) {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

function applyTranslateGemmaTemplate() {
  throw new Error(
    'TranslateGemma template requires structured messages. ' +
    'Use formatChatMessages(messages, "translategemma") instead of applyChatTemplate(prompt, ...).'
  );
}

// Template type to formatter mapping.
// Add new template types here rather than adding switch cases.
const PROMPT_TEMPLATES = {
  'gemma': applyTurnBasedTemplate,
  'llama3': applyHeaderBasedTemplate,
  'gpt-oss': applyChannelBasedTemplate,
  'chatml': applyChatMLTemplate,
  'qwen': applyQwenTemplate,
  'translategemma': applyTranslateGemmaTemplate,
};

export function applyChatTemplate(prompt, templateType) {
  if (templateType == null) {
    return prompt;
  }
  const formatter = PROMPT_TEMPLATES[templateType];
  if (formatter) {
    return formatter(prompt);
  }
  throw new Error(`Unrecognized chat template type: ${templateType}`);
}

// Exports preserved for existing external imports.
export const applyGemmaChatTemplate = applyTurnBasedTemplate;
export const applyLlama3ChatTemplate = applyHeaderBasedTemplate;
export const applyGptOssChatTemplate = applyChannelBasedTemplate;
export const applyQwenChatTemplate = applyQwenTemplate;


export function isStopToken(token, stopTokenIds, eosTokenId) {
  if (stopTokenIds.includes(token)) return true;
  if (typeof eosTokenId === 'number' && token === eosTokenId) return true;
  return false;
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


export function initSpeculativeDecoder(manifest, speculativeConfig) {
  if (!manifest.draftModel) return null;
  if (manifest.draftModel.numTokens == null) {
    throw new Error(`Manifest "${manifest.modelId}" is missing draftModel.numTokens.`);
  }

  return new SpeculativeDecoder({
    numDraftTokens: manifest.draftModel.numTokens,
    maxRejectionRetries: speculativeConfig.maxRejectionRetries,
    enableTreeDraft: speculativeConfig.enableTreeDraft,
    temperature: speculativeConfig.temperature,
    randomSeed: speculativeConfig.randomSeed,
  });
}

// ============================================================================
// QKV Fusion
// ============================================================================


export function fuseQKVWeights(layerWeights, modelConfig) {
  const device = getDevice();
  if (!device) {
    log.debug('QKV Fusion', 'No GPU device, skipping fusion');
    return;
  }

  const { numLayers, numHeads, numKVHeads, headDim, hiddenSize } = modelConfig;
  const qSize = numHeads * headDim;
  const kSize = numKVHeads * headDim;
  const vSize = numKVHeads * headDim;
  const qkvSize = qSize + kSize + vSize;

  const resolveWeight = (value) => {
    if (isWeightBuffer(value)) {
      return {
        buffer: value.buffer,
        dtype: value.dtype ?? null,
        layout: value.layout ?? null,
        shape: Array.isArray(value.shape) ? value.shape : null,
      };
    }
    if (value instanceof GPUBuffer) {
      return {
        buffer: value,
        dtype: getWeightDtype(value),
        layout: null,
        shape: null,
      };
    }
    return null;
  };

  log.debug('QKV Fusion', `Fusing Q/K/V weights for ${numLayers} layers (${qSize}+${kSize}+${vSize}=${qkvSize})`);

  const resolveBytesPerElement = (weight, expectedElements) => {
    const dtype = typeof weight?.dtype === 'string'
      ? weight.dtype.toLowerCase()
      : null;
    if (dtype === 'f16' || dtype === 'bf16') return 2;
    if (dtype === 'f32') return 4;
    const minF16Bytes = expectedElements * 2;
    const minF32Bytes = expectedElements * 4;
    if (weight?.buffer?.size >= minF32Bytes) return 4;
    if (weight?.buffer?.size >= minF16Bytes) return 2;
    return 0;
  };

  let fusedCount = 0;
  for (let l = 0; l < numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (!weights) continue;

    // Skip if already fused or if weights are not GPUBuffers
    if (weights.qkvProj) continue;
    const qProj = resolveWeight(weights.qProj);
    const kProj = resolveWeight(weights.kProj);
    const vProj = resolveWeight(weights.vProj);
    if (!qProj || !kProj || !vProj) {
      continue;
    }

    const qExpectedElements = qSize * hiddenSize;
    const kExpectedElements = kSize * hiddenSize;
    const vExpectedElements = vSize * hiddenSize;
    const bytesPerElement = resolveBytesPerElement(qProj, qExpectedElements);
    const kBytesPerElement = resolveBytesPerElement(kProj, kExpectedElements);
    const vBytesPerElement = resolveBytesPerElement(vProj, vExpectedElements);

    // Pool allocation can round GPUBuffer.size up, so infer logical dtype first and
    // only use buffer size as a minimum-size inference.
    if ((bytesPerElement !== 2 && bytesPerElement !== 4)
      || kBytesPerElement !== bytesPerElement
      || vBytesPerElement !== bytesPerElement) {
      log.debug(
        'QKV Fusion',
        `Layer ${l}: inconsistent projection dtypes (q=${bytesPerElement}, k=${kBytesPerElement}, v=${vBytesPerElement}), skipping`
      );
      continue;
    }

    const normalizedDtype = typeof qProj.dtype === 'string'
      ? qProj.dtype.toLowerCase()
      : null;
    const dtype = normalizedDtype === 'bf16'
      ? 'f16'
      : (
        normalizedDtype === 'f16' || normalizedDtype === 'f32'
          ? normalizedDtype
          : selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement })
      );
    const layout = qProj.layout ?? kProj.layout ?? vProj.layout ?? 'row';
    let fusedShape = [qkvSize, hiddenSize];
    if (Array.isArray(qProj.shape) && qProj.shape.length === 2) {
      if (qProj.shape[0] === qSize && qProj.shape[1] === hiddenSize) {
        fusedShape = [qkvSize, hiddenSize];
      } else if (qProj.shape[1] === qSize && qProj.shape[0] === hiddenSize) {
        fusedShape = [hiddenSize, qkvSize];
      }
    }

    // Create fused QKV buffer: [qkvSize, hiddenSize] row-major
    // Each row is concatenated: [q_row, k_row, v_row]
    const qkvBuffer = device.createBuffer({
      label: `layer_${l}_qkv_proj`,
      size: qkvSize * hiddenSize * bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Copy Q, K, V weights into fused buffer
    // Q: [qSize, hiddenSize] -> offset 0
    // K: [kSize, hiddenSize] -> offset qSize * hiddenSize * bytesPerElement
    // V: [vSize, hiddenSize] -> offset (qSize + kSize) * hiddenSize * bytesPerElement
    const encoder = device.createCommandEncoder({ label: 'qkv_fusion' });
    encoder.copyBufferToBuffer(
      qProj.buffer, 0,
      qkvBuffer, 0,
      qSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      kProj.buffer, 0,
      qkvBuffer, qSize * hiddenSize * bytesPerElement,
      kSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      vProj.buffer, 0,
      qkvBuffer, (qSize + kSize) * hiddenSize * bytesPerElement,
      vSize * hiddenSize * bytesPerElement
    );
    device.queue.submit([encoder.finish()]);

    // Store fused buffer, sizes, and dtype
    weights.qkvProj = createWeightBuffer(
      qkvBuffer,
      dtype,
      layout,
      fusedShape,
      `layer_${l}_qkv_proj`
    );
    weights.qkvSizes = [qSize, kSize, vSize];
    weights.qkvDtype = dtype;
    fusedCount++;
  }

  log.debug('QKV Fusion', `Fused ${fusedCount}/${numLayers} layers`);
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
