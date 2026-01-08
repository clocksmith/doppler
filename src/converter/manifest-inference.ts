import type { ManifestInferenceSchema, PresetSchema } from '../config/schema/index.js';
import { buildRoPEConfig } from './rope-config.js';

/**
 * Detect whether model scales embeddings by sqrt(hiddenSize).
 *
 * Gemma models (and derivatives like FunctionGemma, CodeGemma) scale embeddings.
 * Detection checks preset name, architecture, and model_type for 'gemma'.
 */
export function detectScaleEmbeddings(preset: PresetSchema, config: Record<string, unknown>): boolean {
  // Check preset ID (covers gemma2, gemma3, functiongemma, codegemma if they extend gemma)
  if (preset.id?.toLowerCase().includes('gemma')) return true;

  // Check architecture string from HF config
  const architectures = config.architectures as string[] | undefined;
  if (architectures?.some((arch) => arch.toLowerCase().includes('gemma'))) return true;

  // Check model_type from HF config
  const modelType = config.model_type as string | undefined;
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
export function buildManifestInference(
  preset: PresetSchema,
  config: Record<string, unknown>,
  headDim = 64
): ManifestInferenceSchema {
  const presetInference = preset.inference || {};

  // Build inference config with all required fields explicitly set
  // Use null for "not applicable" - no undefined allowed
  // Note: Some fields like queryPreAttnScalar and scaleEmbeddings are not in
  // InferenceConfigSchema (preset) but ARE in ManifestInferenceSchema, so we
  // read them directly from the HF config or compute them.
  const inference: ManifestInferenceSchema = {
    attention: {
      // queryPreAttnScalar: HF config or compute from headDim (not in preset schema)
      queryPreAttnScalar: config.query_pre_attn_scalar as number ??
        Math.sqrt(headDim),
      attnLogitSoftcapping: presetInference.attention?.attnLogitSoftcapping ??
        config.attn_logit_softcapping as number | null ?? null,
      slidingWindow: presetInference.attention?.slidingWindow ??
        config.sliding_window as number | null ?? null,
      queryKeyNorm: presetInference.attention?.queryKeyNorm ?? false,
    },
    normalization: {
      rmsNormWeightOffset: presetInference.normalization?.rmsNormWeightOffset ?? false,
      postAttentionNorm: presetInference.normalization?.postAttentionNorm ?? false,
      preFeedforwardNorm: presetInference.normalization?.preFeedforwardNorm ?? false,
      postFeedforwardNorm: presetInference.normalization?.postFeedforwardNorm ?? false,
    },
    ffn: {
      activation: (presetInference.ffn?.activation as ManifestInferenceSchema['ffn']['activation']) ?? 'silu',
      gatedActivation: presetInference.ffn?.gatedFFN ?? true,
    },
    rope: buildRoPEConfig(presetInference, config),
    output: {
      finalLogitSoftcapping: presetInference.output?.finalLogitSoftcapping ??
        config.final_logit_softcapping as number | null ?? null,
      tieWordEmbeddings: presetInference.output?.tieWordEmbeddings ??
        config.tie_word_embeddings as boolean ?? false,
      // scaleEmbeddings: detect from model type/architecture
      // Gemma models (and derivatives like FunctionGemma, CodeGemma) scale embeddings by sqrt(hiddenSize)
      // Check: preset name contains 'gemma' OR architecture contains 'gemma' OR model_type contains 'gemma'
      scaleEmbeddings: detectScaleEmbeddings(preset, config),
    },
  };

  // Add layer pattern if defined
  if (presetInference.layerPattern) {
    // Map preset's LayerPatternSchema to manifest's ManifestLayerPatternSchema
    // Preset uses: type ('all_attention' | 'alternating' | 'custom'), globalPattern, globalPatternN
    // Manifest uses: type ('uniform' | 'alternating' | 'every_n'), globalPattern, period
    const presetType = presetInference.layerPattern.type;
    let manifestType: 'uniform' | 'alternating' | 'every_n';
    if (presetType === 'all_attention') {
      manifestType = 'uniform';
    } else if (presetType === 'custom' || (presetType === 'alternating' && presetInference.layerPattern.globalPatternN)) {
      manifestType = 'every_n';
    } else {
      manifestType = 'alternating';
    }

    inference.layerPattern = {
      type: manifestType,
      globalPattern: presetInference.layerPattern.globalPattern as 'odd' | 'even' | undefined,
      period: presetInference.layerPattern.globalPatternN,
    };
  }

  // Add default kernel path based on preset ID and quantization
  // This will be set later when we know the quantization
  inference.defaultKernelPath = presetInference.kernelPath as string | undefined;

  return inference;
}
