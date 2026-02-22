

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
import { selectRuleValue } from '../../../rules/rule-registry.js';


function isRDRRManifest(manifest) {
  return manifest !== null && typeof manifest === 'object' && Array.isArray( (manifest).shards);
}


function resolveQ4KConfig(manifest, kernelPath, kernelPathSource = 'none') {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
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
  const keepF32Weights = getRuntimeConfig().inference.compute.keepF32Weights;

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


export async function initRoPEFrequencies(config, useGPU) {
  const {
    headDim,
    maxSeqLen,
    ropeTheta,
    ropeLocalTheta,
    ropeScale,
    ropeScalingType,
    ropeScaling,
  } = config;
  if (!Number.isFinite(ropeScale) || ropeScale <= 0) {
    throw new Error(`RoPE scale must be a positive number; got "${ropeScale}".`);
  }

  const halfDim = headDim / 2;
  const isYarn = ropeScalingType === 'yarn';

  // Compute global (full_attention) frequencies
  const globalFreqs = computeRoPEFreqsForTheta(
    ropeTheta, headDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling
  );

  // Compute local (sliding_attention) frequencies if different from global.
  // Models with dual RoPE use different theta for local vs global attention layers.
  
  let localFreqs = null;
  if (ropeLocalTheta && ropeLocalTheta !== ropeTheta) {
    localFreqs = computeRoPEFreqsForTheta(
      ropeLocalTheta, headDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling
    );
    log.debug('Pipeline', `Dual RoPE: local theta=${ropeLocalTheta}, global theta=${ropeTheta}`);
  }

  if (isYarn) {
    // Log YARN params (already validated in computeRoPEFreqs)
    log.debug('Pipeline', `YARN RoPE: factor=${ropeScaling?.factor}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}`);
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

    log.debug('Pipeline', `RoPE frequencies initialized (GPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, theta=${ropeTheta}${ropeLocalTheta ? `, localTheta=${ropeLocalTheta}` : ''}, scaling=${isYarn ? 'yarn' : 'linear'}`);

    return {
      cos: cosBuffer,
      sin: sinBuffer,
      localCos: localCosBuffer,
      localSin: localSinBuffer,
    };
  }

  log.debug('Pipeline', `RoPE frequencies initialized (CPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, theta=${ropeTheta}${ropeLocalTheta ? `, localTheta=${ropeLocalTheta}` : ''}, scaling=${isYarn ? 'yarn' : 'linear'}`);

  return {
    cos: globalFreqs.cos,
    sin: globalFreqs.sin,
    localCos: localFreqs?.cos,
    localSin: localFreqs?.sin,
  };
}


export function isGPURoPEBuffers(buffers) {
  return buffers.cos instanceof GPUBuffer;
}

// ============================================================================
// KV Cache Setup
// ============================================================================


export function createKVCache(modelConfig, useGPU, debug = false, runtimeConfig) {
  const runtimeKV = runtimeConfig ?? getRuntimeConfig().inference.kvcache;
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
  if (cacheLayout === 'contiguous' && cacheMaxSeqLen >= PAGED_LAYOUT_SEQ_LEN_THRESHOLD) {
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
    if (cacheLayout !== 'paged' && cacheLayout !== 'tiered') {
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
    pageSize: runtimeKV.pageSize,
  };

  
  let kvCache;

  if (modelConfig.slidingWindow && cacheLayout !== 'paged' && cacheLayout !== 'tiered' && cacheLayout !== 'bdpa') {
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
    const isSliding = kvCache instanceof SlidingWindowKVCache;
    log.debug('Pipeline', `KV cache: type=${kvCache?.constructor?.name || 'unknown'}, kvDtype=${kvCache.kvDtype}, layout=${kvCache.layout}, maxSeqLen=${kvCache.maxSeqLen}, windowSize=${isSliding ? kvCache.windowSize : null}`);
  }

  return kvCache;
}

// ============================================================================
// Tokenizer Setup
// ============================================================================


export async function initTokenizer(manifest, options = {}) {
  const { baseUrl, presetTokenizer } = options;
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(manifest, { baseUrl, presetTokenizer });
  return tokenizer;
}

// ============================================================================
// Weight Loading
// ============================================================================


export async function loadWeights(manifest, modelConfig, options = {}) {
  const { storageContext, onProgress, loadingConfig, baseUrl } = options;
  const verifyHashes = options.verifyHashes
    ?? loadingConfig?.shardCache?.verifyHashes;
  if (verifyHashes == null) {
    throw new Error('runtime.loading.shardCache.verifyHashes is required.');
  }

  const dopplerLoader = getDopplerLoader(loadingConfig);
  dopplerLoader.setQ4KConfig(
    resolveQ4KConfig(
      manifest,
      options.resolvedKernelPath ?? null,
      options.kernelPathSource ?? 'none'
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

  // Configure custom shard loader if provided (Native Bridge)
  if (storageContext?.loadShard) {
    log.debug('Pipeline', 'Using custom shard loader (Native Bridge or external)');
    
    const loadShard = async (index) => {
      const data = await storageContext.loadShard(index);
      return data instanceof Uint8Array ? data : new Uint8Array(data);
    };
    dopplerLoader.setCustomShardLoader(loadShard, { verify: verifyHashes });
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

// Template type to formatter mapping.
// Add new template types here rather than adding switch cases.
const PROMPT_TEMPLATES = {
  'gemma': applyTurnBasedTemplate,
  'llama3': applyHeaderBasedTemplate,
  'gpt-oss': applyChannelBasedTemplate,
  'chatml': applyChatMLTemplate,
  'qwen': applyChatMLTemplate,  // Qwen uses ChatML format
};

export function applyChatTemplate(prompt, templateType) {
  if (templateType == null) {
    return prompt;
  }
  const formatter = PROMPT_TEMPLATES[templateType];
  if (formatter) {
    return formatter(prompt);
  }
  throw new Error(`Unsupported chat template type: ${templateType}`);
}

// Legacy exports for backwards compatibility
export const applyGemmaChatTemplate = applyTurnBasedTemplate;
export const applyLlama3ChatTemplate = applyHeaderBasedTemplate;
export const applyGptOssChatTemplate = applyChannelBasedTemplate;
export const applyQwenChatTemplate = applyChatMLTemplate;


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

  log.debug('QKV Fusion', `Fusing Q/K/V weights for ${numLayers} layers (${qSize}+${kSize}+${vSize}=${qkvSize})`);

  let fusedCount = 0;
  for (let l = 0; l < numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (!weights) continue;

    // Skip if already fused or if weights are not GPUBuffers
    if (weights.qkvProj) continue;
    if (!(weights.qProj instanceof GPUBuffer) ||
        !(weights.kProj instanceof GPUBuffer) ||
        !(weights.vProj instanceof GPUBuffer)) {
      continue;
    }

    // Detect bytes per element from actual buffer size
    // Q buffer should be [qSize, hiddenSize] = qSize * hiddenSize elements
    const qExpectedElements = qSize * hiddenSize;
    const qBufferSize = weights.qProj.size;
    const bytesPerElement = qBufferSize / qExpectedElements;

    // Validate: should be 2 (F16) or 4 (F32)
    if (bytesPerElement !== 2 && bytesPerElement !== 4) {
      log.debug('QKV Fusion', `Layer ${l}: unsupported dtype (${bytesPerElement} bytes/elem), skipping`);
      continue;
    }

    const dtype = selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement });

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
      weights.qProj, 0,
      qkvBuffer, 0,
      qSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      weights.kProj, 0,
      qkvBuffer, qSize * hiddenSize * bytesPerElement,
      kSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      weights.vProj, 0,
      qkvBuffer, (qSize + kSize) * hiddenSize * bytesPerElement,
      vSize * hiddenSize * bytesPerElement
    );
    device.queue.submit([encoder.finish()]);

    // Store fused buffer, sizes, and dtype
    weights.qkvProj = qkvBuffer;
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
    // Dynamically import to avoid loading emulation code when disabled
    const { setSimulatorEnv } = await import('/proto/simulator/env.js');
    const { createEmulationConfig, formatBytes, formatBandwidth } = await import('../../../config/schema/emulation.schema.js');
    const { EmulatedVramStore, detectLocalResources } = await import('../../../storage/emulated-vram.js');
    const { getBufferPool } = await import('../../../memory/buffer-pool.js');
    const { createEmulationContext, isEmulationSupported } = await import('/proto/simulator/index.js');

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

    // Check if emulation is supported
    const supported = await isEmulationSupported();
    if (!supported) {
      log.warn('Pipeline', 'Emulation requested but not supported in this environment');
      return null;
    }

    // Create emulation context
    log.info('Pipeline', `Initializing emulation for ${emulationConfig.targetChip}`);
    const ctx = await createEmulationContext(emulationConfig);

    log.info('Pipeline', `Emulation ready: ${ctx.config.topology.gpuCount} virtual GPUs, timing mode: ${ctx.config.timingMode}`);

    return ctx;
  } catch (err) {
    log.error('Pipeline', `Failed to initialize emulation: ${err.message}`);
    // Graceful fallback - continue without emulation
    return null;
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
