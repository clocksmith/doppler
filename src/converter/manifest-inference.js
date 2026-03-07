import { DEFAULT_MANIFEST_INFERENCE } from '../config/schema/index.js';
import { buildRoPEConfig } from './rope-config.js';

const EMBEDDING_TENSOR_NAMES = [
  'language_model.model.embed_tokens.weight',
  'model.embed_tokens.weight',
  'embed_tokens.weight',
  'token_embd.weight',
  'wte.weight',
  'transformer.wte.weight',
];
const EMBEDDING_TENSOR_PATTERNS = [
  'embed_tokens.weight',
  'token_embd.weight',
  'wte.weight',
  'transformer.wte.weight',
  'word_embeddings',
];


export function inferEmbeddingOutputConfig(tensorLocations) {
  const getLocation = (name) => {
    if (tensorLocations instanceof Map) {
      return tensorLocations.get(name);
    }
    return tensorLocations?.[name];
  };

  const entries = tensorLocations instanceof Map
    ? tensorLocations.entries()
    : Object.entries(tensorLocations ?? {});
  for (const [_name, loc] of entries) {
    if (loc?.role === 'embedding' && loc.shape?.length === 2) {
      const [dim0, dim1] = loc.shape;
      const isGGUFLayout = dim0 < dim1;
      return {
        embeddingTranspose: isGGUFLayout,
        embeddingVocabSize: isGGUFLayout ? dim1 : dim0,
      };
    }
  }

  for (const name of EMBEDDING_TENSOR_NAMES) {
    const loc = getLocation(name);
    if (loc?.shape && loc.shape.length === 2) {
      const [dim0, dim1] = loc.shape;
      const isGGUFLayout = dim0 < dim1;
      return {
        embeddingTranspose: isGGUFLayout,
        embeddingVocabSize: isGGUFLayout ? dim1 : dim0,
      };
    }
  }

  return null;
}


// NOTE: detectScaleEmbeddings removed - use preset.inference.output.scaleEmbeddings instead
// Model-family detection via string matching violates "Manifest as Source of Truth" principle

// Auto-detect normalization config from tensor names.
// Prevents bugs like postFeedforwardNorm=false when weights actually exist.
function detectNormalizationFromTensors(tensorNames) {
  const detected = {};

  // Post-attention norm (sandwich norm pattern)
  if (tensorNames.some(name => /post_attention_layernorm\.weight$/.test(name))) {
    detected.postAttentionNorm = true;
  }

  // Pre-feedforward norm (sandwich norm pattern)
  if (tensorNames.some(name => /pre_feedforward_layernorm\.weight$/.test(name))) {
    detected.preFeedforwardNorm = true;
  }

  // Post-feedforward norm (sandwich norm pattern)
  if (tensorNames.some(name => /post_feedforward_layernorm\.weight$/.test(name))) {
    detected.postFeedforwardNorm = true;
  }

  // Per-head query/key normalization
  if (tensorNames.some(name => /self_attn\.(q_norm|k_norm)\.weight$/.test(name))) {
    detected.queryKeyNorm = true;
  }

  return detected;
}

function detectTieWordEmbeddingsFromTensors(tensorNames) {
  if (!Array.isArray(tensorNames) || tensorNames.length === 0) {
    return null;
  }

  let hasEmbedding = false;
  let hasLmHead = false;

  for (const name of tensorNames) {
    const lower = name.toLowerCase();
    if (!hasEmbedding && EMBEDDING_TENSOR_PATTERNS.some((pattern) => lower.includes(pattern))) {
      hasEmbedding = true;
    }
    if (!hasLmHead) {
      if (lower.includes('lm_head')) {
        hasLmHead = true;
      } else if (lower.endsWith('output.weight') && !lower.includes('attn_')) {
        hasLmHead = true;
      }
    }
    if (hasEmbedding && hasLmHead) break;
  }

  if (hasLmHead) return false;
  if (hasEmbedding) return true;
  return null;
}

function detectCausalAttention(modelConfig) {
  if (!modelConfig || typeof modelConfig !== 'object') {
    return null;
  }

  if (modelConfig.use_bidirectional_attention === true || modelConfig.is_decoder === false) {
    return false;
  }

  if (modelConfig.use_bidirectional_attention === false || modelConfig.is_decoder === true) {
    return true;
  }

  return null;
}

function normalizeLayerTypeName(value) {
  if (typeof value !== 'string') return '';
  return value.trim().toLowerCase();
}

function normalizeCustomLayerType(value) {
  const normalized = normalizeLayerTypeName(value);
  if (!normalized) return null;
  if (
    normalized === 'full_attention'
    || normalized === 'global_attention'
    || normalized === 'full'
    || normalized === 'global'
    || normalized === 'attention'
  ) {
    return 'full_attention';
  }
  if (
    normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'sliding'
    || normalized === 'local'
  ) {
    return 'sliding_attention';
  }
  if (
    normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net'
  ) {
    return 'linear_attention';
  }
  if (
    normalized === 'conv'
    || normalized === 'convolution'
    || normalized === 'liv_conv'
    || normalized === 'liv_convolution'
  ) {
    return 'conv';
  }
  return null;
}

function normalizeLayerTypesForManifest(layerTypes, contextLabel) {
  if (!Array.isArray(layerTypes)) {
    throw new Error(`${contextLabel} requires layerTypes array.`);
  }

  return layerTypes.map((layerType, index) => {
    const normalized = normalizeCustomLayerType(layerType);
    if (!normalized) {
      throw new Error(
        `${contextLabel} has unsupported layerTypes[${index}]="${layerType}". ` +
        'Supported: conv, full_attention, sliding_attention, linear_attention.'
      );
    }
    return normalized;
  });
}

function detectEveryNOffsetFromLayerTypes(layerTypes, period) {
  if (!Array.isArray(layerTypes) || !Number.isFinite(period) || period <= 0) {
    return null;
  }

  const globalLayerIndices = [];
  for (let i = 0; i < layerTypes.length; i++) {
    const normalized = normalizeLayerTypeName(layerTypes[i]);
    const isGlobal = (
      normalized === 'full_attention'
      || normalized === 'global_attention'
      || normalized === 'full'
      || normalized === 'global'
    );
    if (isGlobal) {
      globalLayerIndices.push(i);
    }
  }

  if (globalLayerIndices.length === 0) {
    return null;
  }

  const first = globalLayerIndices[0];
  for (const index of globalLayerIndices) {
    if (((index - first) % period) !== 0) {
      return null;
    }
  }

  return ((first % period) + period) % period;
}

function normalizeEveryNOffset(value, period) {
  if (!Number.isFinite(period) || period <= 0) return null;
  if (!Number.isFinite(value)) return null;
  const raw = Math.trunc(value);
  return ((raw % period) + period) % period;
}

function detectAttentionOutputGate(presetInference, modelConfig, defaults) {
  if (typeof presetInference?.attention?.attentionOutputGate === 'boolean') {
    return presetInference.attention.attentionOutputGate;
  }
  if (typeof modelConfig?.attn_output_gate === 'boolean') {
    return modelConfig.attn_output_gate;
  }

  const modelType = normalizeLayerTypeName(modelConfig?.model_type);
  const hasLinearAttentionLayers = Array.isArray(modelConfig?.layer_types)
    && modelConfig.layer_types.some((entry) => normalizeCustomLayerType(entry) === 'linear_attention');
  if (
    hasLinearAttentionLayers
    && (modelType === 'qwen2' || modelType === 'qwen3_5' || modelType === 'qwen3_5_text')
  ) {
    return true;
  }

  return defaults.attention.attentionOutputGate;
}

function resolveQueryPreAttnScalar(preset, modelConfig, headDim) {
  const explicit = Number(modelConfig?.query_pre_attn_scalar);
  if (Number.isFinite(explicit) && explicit > 0) {
    return explicit;
  }

  const modelType = normalizeLayerTypeName(modelConfig?.model_type);
  const presetId = normalizeLayerTypeName(preset?.id);
  if (modelType.startsWith('qwen') || presetId === 'qwen3') {
    return headDim;
  }

  return Math.sqrt(headDim);
}

// Build normalization config with auto-detection from tensor names.
// Priority: auto-detected > preset > default
function buildNormalizationConfig(presetInference, modelConfig, defaults, tensorNames) {
  const detected = tensorNames ? detectNormalizationFromTensors(tensorNames) : {};

  return {
    rmsNormEps: presetInference.normalization?.rmsNormEps ??
      modelConfig.rms_norm_eps ??
      modelConfig.attentionLayerNormRMSEpsilon ??
      defaults.normalization.rmsNormEps,
    rmsNormWeightOffset: presetInference.normalization?.rmsNormWeightOffset ?? defaults.normalization.rmsNormWeightOffset,
    // For norm flags: auto-detected > preset > default
    postAttentionNorm: detected.postAttentionNorm ?? presetInference.normalization?.postAttentionNorm ?? defaults.normalization.postAttentionNorm,
    preFeedforwardNorm: detected.preFeedforwardNorm ?? presetInference.normalization?.preFeedforwardNorm ?? defaults.normalization.preFeedforwardNorm,
    postFeedforwardNorm: detected.postFeedforwardNorm ?? presetInference.normalization?.postFeedforwardNorm ?? defaults.normalization.postFeedforwardNorm,
  };
}

function normalizeKernelDtype(value) {
  if (!value) return null;
  const lower = String(value).toLowerCase();
  if (lower === 'bf16') return 'f16';
  if (lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'fp32' || lower === 'float32') return 'f32';
  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4' || lower === 'q4km') return 'q4k';
  return lower;
}

function resolveKernelPathFromPreset(presetInference, quantizationInfo, q4kLayout = null) {
  const kernelPaths = presetInference?.kernelPaths;
  if (!kernelPaths) {
    return presetInference?.kernelPath ?? null;
  }

  const weightKey = normalizeKernelDtype(quantizationInfo?.weights);
  const computeKey = normalizeKernelDtype(quantizationInfo?.compute);
  const hasWeightEntry = weightKey != null && Object.prototype.hasOwnProperty.call(kernelPaths, weightKey);
  const entry = hasWeightEntry ? kernelPaths[weightKey] : kernelPaths.default;
  const weightLabel = weightKey ? `.${weightKey}` : '';
  let resolved = null;
  if (entry == null) {
    if (weightKey) {
      throw new Error(
        `Preset kernelPaths${weightLabel} is missing. ` +
        'Add an explicit quantization mapping or kernelPaths.default instead of relying on JS fallbacks.'
      );
    }
    throw new Error(
      'Preset kernelPaths requires quantizationInfo.weights to resolve defaultKernelPath. ' +
      'Add kernelPaths.default or provide explicit quantizationInfo.weights.'
    );
  }

  if (typeof entry === 'string') {
    resolved = entry;
  } else if (entry && computeKey && Object.prototype.hasOwnProperty.call(entry, computeKey)) {
    resolved = entry[computeKey];
  } else if (entry && typeof entry === 'object' && !Array.isArray(entry) && Object.prototype.hasOwnProperty.call(entry, 'default')) {
    resolved = entry.default;
  } else if (entry && typeof entry === 'object' && !Array.isArray(entry) && !computeKey) {
    throw new Error(
      `Preset kernelPaths${weightLabel} requires quantizationInfo.compute ` +
      'to resolve a compute-specific defaultKernelPath.'
    );
  } else if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
    throw new Error(
      `Preset kernelPaths${weightLabel} is missing compute "${computeKey}". ` +
      'Add an explicit compute-specific mapping or default instead of relying on JS fallbacks.'
    );
  } else {
    throw new Error(
      `Preset kernelPaths${weightLabel} must resolve to a string or object.`
    );
  }

  // Column-wise Q4K must be mapped explicitly in preset JSON; JS must not
  // rewrite kernel-path ids to infer policy.
  if (resolved && q4kLayout === 'col' && resolved.includes('-fused-')) {
    throw new Error(
      `Preset kernelPaths${weightKey ? `.${weightKey}` : ''} resolved fused kernel path "${resolved}" ` +
      'for q4k layout "col". Add an explicit dequant kernel path mapping to the preset instead of relying on JS rewrites.'
    );
  }

  return resolved;
}


// Build manifest inference config from preset and HuggingFace config.
// See manifest-inference.d.ts for type signature.
export function buildManifestInference(preset, config, headDim = 64, quantizationInfo = null, tensorNames = null) {
  const defaults = DEFAULT_MANIFEST_INFERENCE;
  const presetInference = preset.inference || {};
  const modelConfig = config?.text_config ?? config ?? {};
  const presetChatTemplate = presetInference.chatTemplate;
  const chatTemplate = typeof presetChatTemplate === 'string'
    ? { type: presetChatTemplate, enabled: true }
    : {
        type: presetChatTemplate?.type ?? null,
        enabled: presetChatTemplate?.enabled ?? (presetChatTemplate?.type != null),
      };

  // Build inference config with all required fields explicitly set
  // Use null for "not applicable" - no undefined allowed
  const detectedTieWordEmbeddings = detectTieWordEmbeddingsFromTensors(tensorNames);
  const detectedCausalAttention = detectCausalAttention(modelConfig);
  const inference = {
    schema: defaults.schema ?? null,
    presetId: preset.id ?? null,
    attention: {
      queryPreAttnScalar: resolveQueryPreAttnScalar(preset, modelConfig, headDim),
      attnLogitSoftcapping: presetInference.attention?.attnLogitSoftcapping ??
        modelConfig.attn_logit_softcapping ?? defaults.attention.attnLogitSoftcapping,
      slidingWindow: presetInference.attention?.slidingWindow ??
        modelConfig.sliding_window ?? defaults.attention.slidingWindow,
      queryKeyNorm: presetInference.attention?.queryKeyNorm ?? defaults.attention.queryKeyNorm,
      attentionOutputGate: detectAttentionOutputGate(presetInference, modelConfig, defaults),
      causal: detectedCausalAttention ?? presetInference.attention?.causal ?? defaults.attention.causal,
      attentionBias: presetInference.attention?.attentionBias ??
        modelConfig.attention_bias ?? defaults.attention.attentionBias,
    },
    normalization: buildNormalizationConfig(presetInference, modelConfig, defaults, tensorNames),
    ffn: {
      activation: presetInference.ffn?.activation ?? defaults.ffn.activation,
      gatedActivation: presetInference.ffn?.gatedActivation ??
        presetInference.ffn?.gatedFFN ?? defaults.ffn.gatedActivation,
      swigluLimit: presetInference.ffn?.swigluLimit ?? modelConfig.swiglu_limit ?? defaults.ffn.swigluLimit,
    },
    rope: buildRoPEConfig(presetInference, modelConfig),
    output: {
      finalLogitSoftcapping: presetInference.output?.finalLogitSoftcapping ??
        modelConfig.final_logit_softcapping ?? defaults.output.finalLogitSoftcapping,
      tieWordEmbeddings: detectedTieWordEmbeddings ??
        presetInference.output?.tieWordEmbeddings ??
        modelConfig.tie_word_embeddings ?? defaults.output.tieWordEmbeddings,
      scaleEmbeddings: presetInference.output?.scaleEmbeddings ?? defaults.output.scaleEmbeddings,
      embeddingTranspose: defaults.output.embeddingTranspose,
      embeddingVocabSize: defaults.output.embeddingVocabSize,
    },
    layerPattern: { ...defaults.layerPattern },
    chatTemplate,
    pipeline: presetInference.pipeline ?? defaults.pipeline,
    sessionDefaults: defaults.sessionDefaults,
    execution: defaults.execution,
  };

  // Add layer pattern if defined
  if (presetInference.layerPattern) {
    const presetPattern = presetInference.layerPattern;
    const presetType = presetPattern.type;
    let manifestType = 'uniform';
    let globalPattern = null;
    let period = null;
    let offset = null;
    let layerTypes = null;

    if (presetType === 'all_attention') {
      manifestType = 'uniform';
    } else if (presetType === 'custom') {
      manifestType = 'custom';
      const customLayerTypes = Array.isArray(modelConfig.layer_types) && modelConfig.layer_types.length > 0
        ? modelConfig.layer_types
        : presetPattern.layerTypes;
      layerTypes = normalizeLayerTypesForManifest(
        customLayerTypes,
        `Preset "${preset.id ?? 'unknown'}" layerPattern`
      );
    } else if (presetType === 'alternating') {
      if (presetPattern.globalPattern === 'every_n') {
        manifestType = 'every_n';
        period = presetPattern.period ?? null;
      } else {
        manifestType = 'alternating';
        globalPattern = presetPattern.globalPattern ?? null;
      }
    } else if (presetType === 'every_n') {
      manifestType = 'every_n';
      period = presetPattern.period ?? null;
    }

    if (manifestType === 'every_n') {
      if (!Number.isFinite(period) || period <= 0) {
        throw new Error(
          `Preset "${preset.id ?? 'unknown'}" layerPattern requires period > 0 for every_n.`
        );
      }
      globalPattern = null;
      offset = (
        detectEveryNOffsetFromLayerTypes(modelConfig.layer_types, period)
        ?? normalizeEveryNOffset(presetPattern.offset, period)
        ?? 0
      );
    } else if (manifestType === 'alternating') {
      if (globalPattern == null) {
        throw new Error(
          `Preset "${preset.id ?? 'unknown'}" layerPattern requires globalPattern for alternating.`
        );
      }
      period = null;
      offset = null;
    } else if (manifestType === 'custom') {
      if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
        throw new Error(
          `Preset "${preset.id ?? 'unknown'}" layerPattern requires non-empty layerTypes for custom pattern.`
        );
      }
      globalPattern = null;
      period = null;
      offset = null;
    }

    inference.layerPattern = {
      type: manifestType,
      globalPattern,
      period,
      offset,
      layerTypes: manifestType === 'custom' ? layerTypes : null,
    };
  }

  // Preserve explicit per-layer metadata when the model config provides layer_types
  // but the preset does not define a layer pattern.
  const hasConfigLayerTypes = Array.isArray(modelConfig.layer_types) && modelConfig.layer_types.length > 0;
  const presetPatternType = presetInference.layerPattern?.type ?? null;
  const shouldPreferConfigLayerTypes = hasConfigLayerTypes && (
    !presetInference.layerPattern
    || presetPatternType === 'all_attention'
    || presetPatternType === 'uniform'
  );
  if (shouldPreferConfigLayerTypes) {
    inference.layerPattern = {
      type: 'custom',
      globalPattern: null,
      period: null,
      offset: null,
      layerTypes: normalizeLayerTypesForManifest(
        modelConfig.layer_types,
        `Model "${preset.id ?? 'unknown'}" config.layer_types`
      ),
    };
  }

  // Add default kernel path based on preset ID, quantization, and q4k layout
  // Layout is now in quantizationInfo.layout: 'row' (fused) or 'col' (dequant)
  const q4kLayout = quantizationInfo?.layout ?? null;
  inference.defaultKernelPath = resolveKernelPathFromPreset(presetInference, quantizationInfo, q4kLayout) ?? defaults.defaultKernelPath;

  return inference;
}
