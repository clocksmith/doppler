

import { parseModelConfig } from './config.js';
import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { KVCache, SlidingWindowKVCache } from '../kv-cache.js';
import { Tokenizer } from '../tokenizer.js';
import { MoERouter } from '../moe-router.js';
import { SpeculativeDecoder } from '../speculative.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { log, setGPUDevice, trace as debugTrace } from '../../debug/index.js';
import { PAGED_LAYOUT_SEQ_LEN_THRESHOLD } from '../../config/schema/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { getActiveKernelPath, getActiveKernelPathSource, isActiveKernelPathFusedQ4K } from '../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../rules/rule-registry.js';


function isRDRRManifest(manifest) {
  return manifest !== null && typeof manifest === 'object' && Array.isArray( (manifest).shards);
}


function resolveQ4KConfig(manifest) {
  const activeKernelPath = getActiveKernelPath();
  const pathSource = getActiveKernelPathSource();
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const q4kLayout =  (manifest?.config)?.q4kLayout;
  const keepF32Weights = getRuntimeConfig().inference.compute.keepF32Weights;

  let useFused = activeKernelPath ? isActiveKernelPathFusedQ4K() : hasSubgroups;
  if (q4kLayout === 'column_wise') {
    useFused = false;
  }

  const pathLabel = activeKernelPath?.id ?? 'auto';
  debugTrace.loader(`Q4K config: fused=${useFused}, kernelPath=${pathLabel}, source=${pathSource}, layout=${q4kLayout ?? 'default'}, subgroups=${hasSubgroups}`);

  return {
    useFusedQ4K: useFused,
    q4kLayout:  (q4kLayout) ?? null,
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
    if (ropeScaling?.beta_fast == null || ropeScaling?.beta_slow == null ||
        ropeScaling?.original_max_position_embeddings == null) {
      throw new Error(
        `RoPE scaling type is 'yarn' but YARN params missing. ` +
        `Manifest must provide beta_fast, beta_slow, and original_max_position_embeddings. ` +
        `Got: beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}, ` +
        `original_max_position_embeddings=${ropeScaling?.original_max_position_embeddings}`
      );
    }
    // Extract validated YARN params (no hidden defaults - all guaranteed non-null)
    const yarnFactor = ropeScaling.factor ?? ropeScale;
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
    ropeScale = 1.0,
    ropeScalingType,
    ropeScaling,
  } = config;

  const halfDim = headDim / 2;
  const isYarn = ropeScalingType === 'yarn';

  // Compute global (full_attention) frequencies
  const globalFreqs = computeRoPEFreqsForTheta(
    ropeTheta, headDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling
  );

  // Compute local (sliding_attention) frequencies if different from global
  // Gemma 3 uses 10K for local layers and 1M for global layers
  
  let localFreqs = null;
  if (ropeLocalTheta && ropeLocalTheta !== ropeTheta) {
    localFreqs = computeRoPEFreqsForTheta(
      ropeLocalTheta, headDim, maxSeqLen, ropeScale, ropeScalingType, ropeScaling
    );
    log.debug('Pipeline', `Dual RoPE: local theta=${ropeLocalTheta}, global theta=${ropeTheta}`);
  }

  if (isYarn) {
    // Log YARN params (already validated in computeRoPEFreqs)
    log.debug('Pipeline', `YARN RoPE: factor=${ropeScaling?.factor ?? ropeScale}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}`);
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

  // Sliding-window attention only needs a bounded KV cache
  if (slidingWindow && Number.isFinite(slidingWindow) && slidingWindow > 0) {
    if (runtimeKV.windowSize > 0) {
      slidingWindow = Math.min(slidingWindow, runtimeKV.windowSize);
    }
    cacheMaxSeqLen = Math.min(cacheMaxSeqLen, slidingWindow);
    cacheLayout = 'contiguous';
  }

  // GPU paged KV cache is not implemented yet
  if (useGPU && cacheLayout === 'paged') {
    const fallbackMaxSeqLen = runtimeKV.gpuPagedFallbackMaxSeqLen;
    cacheMaxSeqLen = Math.min(modelMaxSeqLen, fallbackMaxSeqLen);
    cacheLayout = 'contiguous';
    log.warn('Pipeline', `Paged GPU KV cache not supported. Capping KV cache to ${cacheMaxSeqLen} tokens.`);
  }

  // Use f16 KV cache when supported to reduce VRAM.
  // For attention logit softcapping (e.g., Gemma 2), allow forcing F32 via runtime config
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

  
  const cacheConfig = {
    numLayers: modelConfig.numLayers,
    numHeads: modelConfig.numKVHeads || modelConfig.numHeads,
    headDim: modelConfig.headDim,
    maxSeqLen: cacheMaxSeqLen,
    useGPU,
    layout: cacheLayout,
    kvDtype,
    pageSize: runtimeKV.pageSize,
  };

  
  let kvCache;

  if (modelConfig.slidingWindow) {
    kvCache = new SlidingWindowKVCache({
      ...cacheConfig,
      windowSize: slidingWindow ?? modelConfig.slidingWindow,
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
  dopplerLoader.setQ4KConfig(resolveQ4KConfig(manifest));

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
  const modelId = manifest.modelId || manifest.model_id || 'default';
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


export function applyGemmaChatTemplate(prompt) {
  const userTurn = `<start_of_turn>user\n${prompt}<end_of_turn>\n`;
  const modelTurn = `<start_of_turn>model\n`;
  return userTurn + modelTurn;
}


export function applyLlama3ChatTemplate(prompt) {
  return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n${prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}


export function applyGptOssChatTemplate(prompt) {
  return `<|start|>user<|message|>${prompt}<|end|><|start|>assistant<|channel|>final<|message|>`;
}

export function applyChatMLTemplate(prompt) {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}


export function applyQwenChatTemplate(prompt) {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}


export function applyChatTemplate(prompt, templateType) {
  switch (templateType) {
    case 'gemma':
      return applyGemmaChatTemplate(prompt);
    case 'llama3':
      return applyLlama3ChatTemplate(prompt);
    case 'gpt-oss':
      return applyGptOssChatTemplate(prompt);
    case 'chatml':
      return applyChatMLTemplate(prompt);
    case 'qwen':
      return applyQwenChatTemplate(prompt);
    case null:
    case undefined:
      return prompt;
    default:
      throw new Error(`Unsupported chat template type: ${templateType}`);
  }
}


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
    // Q: [qSize, hiddenSize] → offset 0
    // K: [kSize, hiddenSize] → offset qSize * hiddenSize * bytesPerElement
    // V: [vSize, hiddenSize] → offset (qSize + kSize) * hiddenSize * bytesPerElement
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

/**
 * Initialize NVIDIA superchip emulation if enabled in runtime config.
 *
 * This creates an EmulationContext that provides virtual GPUs, CPUs,
 * and interconnect simulation for testing distributed inference patterns.
 *
 * @param {import('../../config/schema/doppler.schema.js').RuntimeConfigSchema} runtimeConfig
 * @returns {Promise<import('../../simulator/index.js').EmulationContext|null>}
 */
export async function initEmulation(runtimeConfig) {
  const emulationConfig = runtimeConfig?.emulation;

  // Skip if emulation is not enabled
  if (!emulationConfig?.enabled) {
    return null;
  }

  try {
    // Dynamically import to avoid loading emulation code when disabled
    const { createEmulationContext, isEmulationSupported } = await import('../../simulator/index.js');

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

/**
 * Destroy emulation context and clean up resources.
 *
 * @param {import('../../simulator/index.js').EmulationContext|null} emulation
 */
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
