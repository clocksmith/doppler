import { buildRoPEConfig } from './rope-config.js';

const EMBEDDING_TENSOR_NAMES = [
  'language_model.model.embed_tokens.weight',
  'model.embed_tokens.weight',
  'embed_tokens.weight',
  'token_embd.weight',
  'wte.weight',
  'transformer.wte.weight',
];


export function inferEmbeddingOutputConfig(tensorLocations) {
  const getLocation = (name) => {
    if (tensorLocations instanceof Map) {
      return tensorLocations.get(name);
    }
    return tensorLocations?.[name];
  };

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


export function detectScaleEmbeddings(preset, config) {
  // Check preset ID (covers gemma2, gemma3, functiongemma, codegemma if they extend gemma)
  if (preset.id?.toLowerCase().includes('gemma')) return true;

  // Check architecture string from HF config
  const architectures = config.architectures;
  if (architectures?.some((arch) => arch.toLowerCase().includes('gemma'))) return true;

  // Check model_type from HF config
  const modelType = config.model_type;
  if (modelType?.toLowerCase().includes('gemma')) return true;

  return false;
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

function resolveKernelPathFromPreset(presetInference, quantizationInfo) {
  const kernelPaths = presetInference?.kernelPaths;
  if (!kernelPaths) {
    return presetInference?.kernelPath ?? null;
  }

  const weightKey = normalizeKernelDtype(quantizationInfo?.weights);
  const computeKey = normalizeKernelDtype(quantizationInfo?.compute) ?? (quantizationInfo ? 'f16' : null);

  const entry = (weightKey && kernelPaths[weightKey]) || kernelPaths.default;
  if (typeof entry === 'string') {
    return entry;
  }
  if (entry && computeKey && entry[computeKey]) {
    return entry[computeKey];
  }
  if (entry && entry.default) {
    return entry.default;
  }
  return presetInference?.kernelPath ?? null;
}


export function buildManifestInference(preset, config, headDim = 64, quantizationInfo = null) {
  const presetInference = preset.inference || {};
  const presetChatTemplate = presetInference.chatTemplate;
  const chatTemplate = typeof presetChatTemplate === 'string'
    ? { type: presetChatTemplate, enabled: true }
    : {
        type: presetChatTemplate?.type ?? null,
        enabled: presetChatTemplate?.enabled ?? (presetChatTemplate?.type != null),
      };

  // Build inference config with all required fields explicitly set
  // Use null for "not applicable" - no undefined allowed
  const inference = {
    attention: {
      queryPreAttnScalar: config.query_pre_attn_scalar ?? Math.sqrt(headDim),
      attnLogitSoftcapping: presetInference.attention?.attnLogitSoftcapping ??
        config.attn_logit_softcapping ?? null,
      slidingWindow: presetInference.attention?.slidingWindow ??
        config.sliding_window ?? null,
      queryKeyNorm: presetInference.attention?.queryKeyNorm ?? false,
    },
    normalization: {
      rmsNormWeightOffset: presetInference.normalization?.rmsNormWeightOffset ?? false,
      postAttentionNorm: presetInference.normalization?.postAttentionNorm ?? false,
      preFeedforwardNorm: presetInference.normalization?.preFeedforwardNorm ?? false,
      postFeedforwardNorm: presetInference.normalization?.postFeedforwardNorm ?? false,
    },
    ffn: {
      activation: presetInference.ffn?.activation ?? 'silu',
      gatedActivation: presetInference.ffn?.gatedActivation ?? presetInference.ffn?.gatedFFN ?? true,
      swigluLimit: presetInference.ffn?.swigluLimit ?? config.swiglu_limit ?? null,
    },
    rope: buildRoPEConfig(presetInference, config),
    output: {
      finalLogitSoftcapping: presetInference.output?.finalLogitSoftcapping ??
        config.final_logit_softcapping ?? null,
      tieWordEmbeddings: presetInference.output?.tieWordEmbeddings ??
        config.tie_word_embeddings ?? false,
      scaleEmbeddings: detectScaleEmbeddings(preset, config),
      embeddingTranspose: false,
      embeddingVocabSize: null,
    },
    chatTemplate,
  };

  // Add layer pattern if defined
  if (presetInference.layerPattern) {
    const presetType = presetInference.layerPattern.type;
    let manifestType;
    if (presetType === 'all_attention') {
      manifestType = 'uniform';
    } else if (presetType === 'custom' || (presetType === 'alternating' && presetInference.layerPattern.globalPatternN)) {
      manifestType = 'every_n';
    } else {
      manifestType = 'alternating';
    }

    inference.layerPattern = {
      type: manifestType,
      globalPattern: presetInference.layerPattern.globalPattern,
      period: presetInference.layerPattern.globalPatternN,
    };
  }

  // Add default kernel path based on preset ID and quantization
  inference.defaultKernelPath = resolveKernelPathFromPreset(presetInference, quantizationInfo);

  return inference;
}
