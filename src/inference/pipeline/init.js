/**
 * Pipeline initialization - model loading, tokenizer setup, KV cache, RoPE.
 *
 * This module handles all initialization tasks for the inference pipeline:
 * - Loading model manifest and parsing configuration
 * - Initializing tokenizer
 * - Setting up KV cache (standard or sliding window)
 * - Computing RoPE frequency buffers (linear or YARN scaling)
 * - Loading model weights via DopplerLoader
 * - Setting up MoE router if applicable
 *
 * @module inference/pipeline/init
 */

import { parseModelConfig } from './config.js';
import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer } from '../../gpu/buffer-pool.js';
import { KVCache, SlidingWindowKVCache } from '../kv-cache.js';
import { Tokenizer } from '../tokenizer.js';
import { MoERouter } from '../moe-router.js';
import { SpeculativeDecoder } from '../speculative.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { log, setGPUDevice, trace as debugTrace } from '../../debug/index.js';
import { PAGED_LAYOUT_SEQ_LEN_THRESHOLD } from '../../config/schema/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { getActiveKernelPath, getActiveKernelPathSource, isActiveKernelPathFusedQ4K } from '../../config/kernel-path-loader.js';

/**
 * @param {unknown} manifest
 * @returns {manifest is import('../../storage/rdrr-format.js').RDRRManifest}
 */
function isRDRRManifest(manifest) {
  return manifest !== null && typeof manifest === 'object' && Array.isArray(/** @type {any} */ (manifest).shards);
}

/**
 * Resolve Q4K load configuration from runtime/kernel/manifest inputs.
 *
 * @param {import('./config.js').Manifest} manifest
 * @returns {import('../../loader/loader-types.js').Q4KConfig}
 */
function resolveQ4KConfig(manifest) {
  const activeKernelPath = getActiveKernelPath();
  const pathSource = getActiveKernelPathSource();
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const q4kLayout = /** @type {{ q4kLayout?: string } | undefined} */ (manifest?.config)?.q4kLayout;
  const keepF32Weights = getRuntimeConfig().inference.compute.keepF32Weights;

  let useFused = activeKernelPath ? isActiveKernelPathFusedQ4K() : hasSubgroups;
  if (typeof window !== 'undefined' && /** @type {{ DOPPLER_DISABLE_FUSED_Q4K?: boolean }} */ (/** @type {unknown} */ (window)).DOPPLER_DISABLE_FUSED_Q4K) {
    useFused = false;
  }
  if (q4kLayout === 'column_wise') {
    useFused = false;
  }

  const pathLabel = activeKernelPath?.id ?? 'auto';
  debugTrace.loader(`Q4K config: fused=${useFused}, kernelPath=${pathLabel}, source=${pathSource}, layout=${q4kLayout ?? 'default'}, subgroups=${hasSubgroups}`);

  return {
    useFusedQ4K: useFused,
    q4kLayout: /** @type {'flat' | 'row_wise' | 'column_wise'} */ (q4kLayout) ?? null,
    keepF32Weights,
  };
}

// ============================================================================
// RoPE Initialization
// ============================================================================

/**
 * Compute RoPE cos/sin frequencies for a given theta.
 * Internal helper for initRoPEFrequencies.
 *
 * @param {number} theta
 * @param {number} headDim
 * @param {number} maxSeqLen
 * @param {number} ropeScale
 * @param {string | undefined} ropeScalingType
 * @param {import('./init.js').RoPEConfig['ropeScaling']} ropeScaling
 * @returns {{ cos: Float32Array; sin: Float32Array }}
 */
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

/**
 * Initialize RoPE (Rotary Position Embedding) frequency buffers.
 *
 * Supports:
 * - Linear scaling: Uniform scaling across all dimensions
 * - YARN (Yet Another RoPE eNhancement): Per-dimension scaling based on wavelength
 * - Dual theta: Different RoPE theta for local vs global attention (Gemma 3)
 *
 * @param {import('./init.js').RoPEConfig} config - RoPE configuration
 * @param {boolean} useGPU - Whether to upload to GPU
 * @returns {Promise<import('./init.js').RoPEBuffers>} RoPE frequency buffers (cos and sin, plus localCos/localSin if ropeLocalTheta differs)
 */
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
  /** @type {{ cos: Float32Array; sin: Float32Array } | null} */
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

    /** @type {GPUBuffer | undefined} */
    let localCosBuffer;
    /** @type {GPUBuffer | undefined} */
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

/**
 * Type guard to check if RoPE buffers are GPU buffers.
 *
 * @param {import('./init.js').RoPEBuffers} buffers
 * @returns {buffers is { cos: GPUBuffer; sin: GPUBuffer; localCos?: GPUBuffer; localSin?: GPUBuffer }}
 */
export function isGPURoPEBuffers(buffers) {
  return buffers.cos instanceof GPUBuffer;
}

// ============================================================================
// KV Cache Setup
// ============================================================================

/**
 * Create and configure KV cache based on model configuration.
 *
 * @param {import('./config.js').ParsedModelConfig} modelConfig - Parsed model configuration
 * @param {boolean} useGPU - Whether GPU is available
 * @param {boolean} [debug] - Debug mode flag
 * @param {import('../../config/schema/index.js').KVCacheConfigSchema} [runtimeConfig]
 * @returns {import('../kv-cache.js').KVCache | import('../kv-cache.js').SlidingWindowKVCache}
 */
export function createKVCache(modelConfig, useGPU, debug = false, runtimeConfig) {
  const runtimeKV = runtimeConfig ?? getRuntimeConfig().inference.kvcache;
  const modelMaxSeqLen = modelConfig.maxSeqLen ?? runtimeKV.maxSeqLen;
  let slidingWindow = Number(modelConfig.slidingWindow || 0) || null;

  let cacheMaxSeqLen = modelMaxSeqLen;
  if (Number.isFinite(runtimeKV.maxSeqLen) && runtimeKV.maxSeqLen > 0) {
    cacheMaxSeqLen = Math.min(cacheMaxSeqLen, runtimeKV.maxSeqLen);
  }

  /** @type {'contiguous' | 'paged'} */
  let cacheLayout = runtimeKV.layout ?? (cacheMaxSeqLen > PAGED_LAYOUT_SEQ_LEN_THRESHOLD ? 'paged' : 'contiguous');

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
  /** @type {'f16' | 'f32'} */
  let kvDtype = runtimeKV.kvDtype;
  if (kvDtype === 'f16' && (!useGPU || !gpuCaps.hasF16)) {
    kvDtype = 'f32';
  }
  if (forceF32KV) {
    kvDtype = 'f32';
  }
  if (forceF32KV && debug) {
    log.debug('Pipeline', `Forcing F32 KV cache (attnLogitSoftcapping=${modelConfig.attnLogitSoftcapping}, forceF32Softcap=true)`);
  }

  /** @type {import('./init.js').KVCacheConfig} */
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

  /** @type {import('../kv-cache.js').KVCache | import('../kv-cache.js').SlidingWindowKVCache} */
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

/**
 * Initialize tokenizer from manifest.
 *
 * @param {import('./config.js').Manifest & import('../tokenizer.js').ModelManifest} manifest - Model manifest
 * @param {string} [baseUrl] - Base URL for loading tokenizer.json
 * @returns {Promise<import('../tokenizer.js').Tokenizer>}
 */
export async function initTokenizer(manifest, baseUrl) {
  const tokenizer = new Tokenizer();
  await tokenizer.initialize(manifest, { baseUrl });
  return tokenizer;
}

// ============================================================================
// Weight Loading
// ============================================================================

/**
 * Load model weights via DopplerLoader.
 *
 * @param {import('./config.js').Manifest} manifest - Model manifest
 * @param {import('./config.js').ParsedModelConfig} modelConfig - Parsed model configuration
 * @param {import('./init.js').LoadWeightsOptions} [options] - Load options
 * @returns {Promise<import('./init.js').WeightLoadResult>}
 */
export async function loadWeights(manifest, modelConfig, options = {}) {
  const { storageContext, onProgress, verifyHashes = false, loadingConfig, baseUrl } = options;

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
    /**
     * @param {number} index
     * @returns {Promise<Uint8Array>}
     */
    const loadShard = async (index) => {
      const data = await storageContext.loadShard(index);
      return data instanceof Uint8Array ? data : new Uint8Array(data);
    };
    dopplerLoader.setCustomShardLoader(loadShard, {
      verify: true,
    });
    if (isRDRRManifest(manifest)) {
      dopplerLoader.setManifest(manifest);
    }
  }

  await dopplerLoader.init();

  // Load model via DopplerLoader
  // Skip hash verification by default - verification happens during download
  const modelId = manifest.modelId || manifest.model_id || 'default';
  await dopplerLoader.load(modelId, {
    verifyHashes: storageContext?.loadShard ? false : verifyHashes,
    onProgress: onProgress || ((info) => {
      // Shard and layer progress are logged by loader with source info
      if (info.stage !== 'layers' && info.stage !== 'shards') {
        log.verbose('Loader', `${info.stage}: ${Math.round(info.progress * 100)}%`);
      }
    }),
  });

  // Map layer weights
  /** @type {Map<string, import('./types.js').LayerWeights>} */
  const layerWeights = new Map();
  for (let l = 0; l < modelConfig.numLayers; l++) {
    const weights = dopplerLoader.getLayerWeights(l);
    if (weights) {
      layerWeights.set(`layer_${l}`, weights);
    }
  }

  // Collect per-layer router weights for MoE
  /** @type {Map<number, import('./types.js').RouterWeights>} */
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

/**
 * Apply Gemma chat template to a prompt.
 *
 * Format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
 * Note: BOS token (2) is added by tokenizer.
 *
 * @param {string} prompt - Raw user prompt
 * @returns {string} Formatted prompt with chat template
 */
export function applyGemmaChatTemplate(prompt) {
  const userTurn = `<start_of_turn>user\n${prompt}<end_of_turn>\n`;
  const modelTurn = `<start_of_turn>model\n`;
  return userTurn + modelTurn;
}

/**
 * Apply Llama 3 chat template to a prompt.
 *
 * Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
 * Note: BOS token is included as <|begin_of_text|>
 *
 * @param {string} prompt - Raw user prompt
 * @returns {string} Formatted prompt with chat template
 */
export function applyLlama3ChatTemplate(prompt) {
  return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n${prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}

/**
 * Apply GPT-OSS chat template to a prompt.
 *
 * Format: <|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>
 *
 * The model expects specific channels (analysis, commentary, final).
 * For basic completion, we use the 'final' channel which produces visible output.
 *
 * @param {string} prompt - Raw user prompt
 * @returns {string} Formatted prompt with chat template
 */
export function applyGptOssChatTemplate(prompt) {
  return `<|start|>user<|message|>${prompt}<|end|><|start|>assistant<|channel|>final<|message|>`;
}

/**
 * Apply chat template based on template type from config.
 * This is the config-driven entry point for chat templates.
 *
 * @param {string} prompt - Raw user prompt
 * @param {string | null | undefined} templateType - Template type from preset config ('gemma', 'llama3', 'gpt-oss', or null)
 * @returns {string} Formatted prompt (or original if no template type)
 */
export function applyChatTemplate(prompt, templateType) {
  switch (templateType) {
    case 'gemma':
      return applyGemmaChatTemplate(prompt);
    case 'llama3':
      return applyLlama3ChatTemplate(prompt);
    case 'gpt-oss':
      return applyGptOssChatTemplate(prompt);
    default:
      return prompt;
  }
}

/**
 * Check if a token is a stop token.
 *
 * @param {number} token - Token ID to check
 * @param {number[]} stopTokenIds - Configured stop token IDs
 * @param {number} [eosTokenId] - EOS token from tokenizer
 * @returns {boolean} True if token should stop generation
 */
export function isStopToken(token, stopTokenIds, eosTokenId) {
  if (stopTokenIds.includes(token)) return true;
  if (typeof eosTokenId === 'number' && token === eosTokenId) return true;
  return false;
}

// ============================================================================
// MoE Router Setup
// ============================================================================

/**
 * Initialize MoE router if model uses Mixture of Experts.
 *
 * @param {import('./config.js').ParsedModelConfig} modelConfig - Parsed model configuration
 * @param {Map<string, import('./types.js').LayerWeights>} layerWeights - Layer weights map
 * @returns {import('../moe-router.js').MoERouter | null} MoE router or null if not MoE model
 */
export function initMoERouter(modelConfig, layerWeights) {
  if (!modelConfig.useMoE) return null;

  const router = new MoERouter({
    numExperts: modelConfig.numExperts,
    topK: modelConfig.moeTopK || 2,
    hiddenSize: modelConfig.hiddenSize,
    normalizeWeights: true,
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

/**
 * Initialize speculative decoder if draft model is available.
 *
 * @param {import('./config.js').Manifest} manifest - Model manifest
 * @returns {import('../speculative.js').SpeculativeDecoder | null} Speculative decoder or null if no draft model
 */
export function initSpeculativeDecoder(manifest) {
  if (!manifest.draftModel) return null;

  return new SpeculativeDecoder({
    numDraftTokens: manifest.draftModel.numTokens || 5,
  });
}

// ============================================================================
// QKV Fusion
// ============================================================================

/**
 * Fuse Q/K/V projection weights into a single QKV weight for optimized inference.
 *
 * This enables 3→1 matmul fusion: instead of 3 separate matmuls for Q, K, V projections,
 * we do one larger matmul and split the output. This saves 2 dispatch barriers.
 *
 * @param {Map<string, import('./types.js').LayerWeights>} layerWeights - Layer weights map
 * @param {import('./config.js').ParsedModelConfig} modelConfig - Parsed model configuration
 */
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

    const dtype = bytesPerElement === 2 ? 'f16' : 'f32';

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
