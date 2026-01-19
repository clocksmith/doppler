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
    normalization: {
      rmsNormEps: presetInference.normalization?.rmsNormEps ??
        modelConfig.rms_norm_eps ??
        modelConfig.attentionLayerNormRMSEpsilon ??
        defaults.normalization.rmsNormEps,
      rmsNormWeightOffset: presetInference.normalization?.rmsNormWeightOffset ?? defaults.normalization.rmsNormWeightOffset,
      postAttentionNorm: presetInference.normalization?.postAttentionNorm ?? defaults.normalization.postAttentionNorm,
      preFeedforwardNorm: presetInference.normalization?.preFeedforwardNorm ?? defaults.normalization.preFeedforwardNorm,
      postFeedforwardNorm: presetInference.normalization?.postFeedforwardNorm ?? defaults.normalization.postFeedforwardNorm,
    },
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
      tieWordEmbeddings: presetInference.output?.tieWordEmbeddings ??
        modelConfig.tie_word_embeddings ?? defaults.output.tieWordEmbeddings,
      scaleEmbeddings: detectScaleEmbeddings(preset, config),
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

  // Add default kernel path based on preset ID and quantization
  inference.defaultKernelPath = resolveKernelPathFromPreset(presetInference, quantizationInfo) ?? defaults.defaultKernelPath;

  return inference;
}
