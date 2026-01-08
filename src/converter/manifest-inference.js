import { buildRoPEConfig } from './rope-config.js';

/**
 * Detect whether model scales embeddings by sqrt(hiddenSize).
 *
 * Gemma models (and derivatives like FunctionGemma, CodeGemma) scale embeddings.
 * Detection checks preset name, architecture, and model_type for 'gemma'.
 */
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

/**
 * Build ManifestInferenceSchema from resolved preset.
 *
 * Extracts inference configuration from the preset and maps it to the
 * manifest schema format. This embeds all model-specific inference
 * parameters in the manifest at conversion time.
 */
export function buildManifestInference(preset, config, headDim = 64) {
  const presetInference = preset.inference || {};

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
      gatedActivation: presetInference.ffn?.gatedFFN ?? true,
    },
    rope: buildRoPEConfig(presetInference, config),
    output: {
      finalLogitSoftcapping: presetInference.output?.finalLogitSoftcapping ??
        config.final_logit_softcapping ?? null,
      tieWordEmbeddings: presetInference.output?.tieWordEmbeddings ??
        config.tie_word_embeddings ?? false,
      scaleEmbeddings: detectScaleEmbeddings(preset, config),
    },
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
  inference.defaultKernelPath = presetInference.kernelPath;

  return inference;
}
