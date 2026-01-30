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
  const computeKey = normalizeKernelDtype(quantizationInfo?.compute) ?? (quantizationInfo ? 'f16' : null);

  const entry = (weightKey && kernelPaths[weightKey]) || kernelPaths.default;
  let resolved = null;
  if (typeof entry === 'string') {
    resolved = entry;
  } else if (entry && computeKey && entry[computeKey]) {
    resolved = entry[computeKey];
  } else if (entry && entry.default) {
    resolved = entry.default;
  } else {
    resolved = presetInference?.kernelPath ?? null;
  }

  // When q4kLayout is 'col' (column-wise), fused Q4K kernels cannot be used.
  // Try to find a corresponding dequant kernel path.
  if (resolved && q4kLayout === 'col' && resolved.includes('-fused-')) {
    const dequantPath = resolved.replace('-fused-', '-dequant-');
    // Return dequant variant (caller should verify it exists)
    return dequantPath;
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
  const inference = {
    presetId: preset.id ?? null,
    attention: {
      queryPreAttnScalar: modelConfig.query_pre_attn_scalar ?? Math.sqrt(headDim),
      attnLogitSoftcapping: presetInference.attention?.attnLogitSoftcapping ??
        modelConfig.attn_logit_softcapping ?? defaults.attention.attnLogitSoftcapping,
      slidingWindow: presetInference.attention?.slidingWindow ??
        modelConfig.sliding_window ?? defaults.attention.slidingWindow,
      queryKeyNorm: presetInference.attention?.queryKeyNorm ?? defaults.attention.queryKeyNorm,
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
  };

  // Add layer pattern if defined
  if (presetInference.layerPattern) {
    const presetPattern = presetInference.layerPattern;
    const presetType = presetPattern.type;
    let manifestType = 'uniform';
    let globalPattern = null;
    let period = null;

    if (presetType === 'all_attention' || presetType === 'custom') {
      manifestType = 'uniform';
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
    } else if (manifestType === 'alternating') {
      if (globalPattern == null) {
        throw new Error(
          `Preset "${preset.id ?? 'unknown'}" layerPattern requires globalPattern for alternating.`
        );
      }
      period = null;
    }

    inference.layerPattern = {
      type: manifestType,
      globalPattern,
      period,
    };
  }

  // Add default kernel path based on preset ID, quantization, and q4k layout
  // Layout is now in quantizationInfo.layout: 'row' (fused) or 'col' (dequant)
  const q4kLayout = quantizationInfo?.layout ?? null;
  inference.defaultKernelPath = resolveKernelPathFromPreset(presetInference, quantizationInfo, q4kLayout) ?? defaults.defaultKernelPath;

  return inference;
}
