/**
 * Model configuration parsing and normalization.
 * Handles HuggingFace, GGUF, and llama.cpp config formats.
 *
 * Architecture: Manifest-First Config Resolution
 * - manifest.inference is the source of truth (populated by converter)
 * - mergeConfig() merges manifest with runtime overrides
 * - toParsedConfigFromMerged() adapts MergedConfig to ParsedModelConfig
 *
 * See: config/merge.ts, config/schema/manifest.schema.ts
 */

import { log } from '../../debug/index.js';
import { DEFAULT_MAX_POSITION_EMBEDDINGS, DEFAULT_RMS_NORM_EPS } from '../../config/schema/index.js';
import { mergeConfig } from '../../config/merge.js';

// =============================================================================
// Model Detection Functions
// =============================================================================

/**
 * @param {import('./config.js').RawConfig} config
 * @param {import('./config.js').Manifest} manifest
 * @returns {number[]}
 */
export function getStopTokenIds(config, manifest) {
  // Priority: manifest.eos_token_id > config.eos_token_id > config.text_config.eos_token_id
  // Model-specific fallbacks are NOT allowed - converter must set eos_token_id
  const eosTokenId = manifest?.eos_token_id ?? config?.eos_token_id ?? config?.text_config?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  return [];
}

// =============================================================================
// Tensor Inference Functions
// =============================================================================

/**
 * @param {import('./config.js').Manifest} manifest
 * @param {number} hiddenSize
 * @param {number | null} [knownNumHeads]
 * @returns {import('./config.js').AttentionParams | null}
 */
export function inferAttentionParams(manifest, hiddenSize, knownNumHeads = null) {
  const tensors = manifest?.tensors ?? {};

  /** @type {number[] | undefined} */
  let qShape;
  /** @type {number[] | undefined} */
  let kShape;

  for (const [name, tensor] of Object.entries(tensors)) {
    const lower = name.toLowerCase();
    if (lower.includes('q_proj') || lower.includes('self_attn.q') || lower.includes('attn_q.weight')) {
      qShape = tensor?.shape;
    }
    if (lower.includes('k_proj') || lower.includes('self_attn.k') || lower.includes('attn_k.weight')) {
      kShape = tensor?.shape;
    }
    if (qShape && kShape) break;
  }

  if (!qShape || !kShape) return null;

  const qOutDim = qShape[0] === hiddenSize ? qShape[1] : qShape[0];
  const kOutDim = kShape[0] === hiddenSize ? kShape[1] : kShape[0];

  if (knownNumHeads && qOutDim % knownNumHeads === 0) {
    const headDim = qOutDim / knownNumHeads;
    if (kOutDim % headDim === 0) {
      const numKVHeads = kOutDim / headDim;
      if (numKVHeads > 0 && knownNumHeads >= numKVHeads) {
        return { numHeads: knownNumHeads, numKVHeads, headDim };
      }
    }
  }

  // Try q_norm weight for headDim
  for (const [name, tensor] of Object.entries(tensors)) {
    if ((name.includes('q_norm') || name.includes('attn_q_norm')) && tensor?.shape?.length === 1) {
      const normHeadDim = tensor.shape[0];
      if (qOutDim % normHeadDim === 0 && kOutDim % normHeadDim === 0) {
        const numHeads = qOutDim / normHeadDim;
        const numKVHeads = kOutDim / normHeadDim;
        if (numHeads >= numKVHeads && numHeads > 0 && numKVHeads > 0) {
          return { numHeads, numKVHeads, headDim: normHeadDim };
        }
      }
    }
  }

  // Try common headDim values
  for (const testHeadDim of [256, 128, 64, 96, 80, 160]) {
    if (qOutDim % testHeadDim === 0 && kOutDim % testHeadDim === 0) {
      const numHeads = qOutDim / testHeadDim;
      const numKVHeads = kOutDim / testHeadDim;
      if (numHeads >= numKVHeads && numHeads > 0 && numKVHeads > 0) {
        return { numHeads, numKVHeads, headDim: testHeadDim };
      }
    }
  }

  // Fallback
  const fallbackHeadDim = Math.floor(hiddenSize / 32);
  if (qOutDim % fallbackHeadDim === 0 && kOutDim % fallbackHeadDim === 0) {
    return {
      numHeads: qOutDim / fallbackHeadDim,
      numKVHeads: kOutDim / fallbackHeadDim,
      headDim: fallbackHeadDim,
    };
  }

  return null;
}

/**
 * @param {import('./config.js').Manifest} manifest
 * @returns {number | null}
 */
export function inferVocabSize(manifest) {
  const tensors = manifest?.tensors ?? {};

  for (const [name, tensor] of Object.entries(tensors)) {
    const lower = name.toLowerCase();
    const isEmbedding =
      lower.includes('embed_tokens.weight') ||
      lower.endsWith('wte.weight') ||
      lower.endsWith('tok_embeddings.weight') ||
      lower.endsWith('word_embeddings.weight') ||
      lower.endsWith('token_embd.weight');
    const isLmHead = lower.includes('lm_head.weight') || lower.endsWith('output.weight');

    if (!isEmbedding && !isLmHead) continue;

    const shape = tensor?.shape;
    if (!Array.isArray(shape) || shape.length === 0) continue;

    const vocabSize = Math.max(...shape);
    if (vocabSize > 1000) return vocabSize;
  }

  return null;
}

// =============================================================================
// Manifest-First Config Resolution (NEW)
// =============================================================================

/**
 * Check if manifest has inference config for manifest-first parsing.
 * @param {import('./config.js').Manifest} manifest
 * @returns {manifest is import('./config.js').Manifest & { inference: import('../../config/schema/index.js').ManifestInferenceSchema }}
 */
export function hasManifestInference(manifest) {
  return 'inference' in manifest && manifest.inference != null;
}

/**
 * Validate required inference fields are present in merged config.
 * Throws if any required field is missing/undefined.
 *
 * @param {import('../../config/merge.js').MergedConfig['inference']} inf
 * @param {string} modelId
 */
function validateRequiredInferenceFields(inf, modelId) {
  /** @type {string[]} */
  const errors = [];

  // Attention fields - non-nullable required
  if (inf.attention.queryPreAttnScalar == null) {
    errors.push('attention.queryPreAttnScalar is required');
  }
  if (inf.attention.queryKeyNorm == null) {
    errors.push('attention.queryKeyNorm is required');
  }
  // Attention fields - nullable required (undefined = missing, null = disabled)
  if (inf.attention.slidingWindow === undefined) {
    errors.push('attention.slidingWindow must be explicitly set (null for no sliding window, or number)');
  }
  if (inf.attention.attnLogitSoftcapping === undefined) {
    errors.push('attention.attnLogitSoftcapping must be explicitly set (null for no softcapping, or number)');
  }

  // Normalization fields
  if (inf.normalization.rmsNormWeightOffset == null) {
    errors.push('normalization.rmsNormWeightOffset is required');
  }
  if (inf.normalization.postAttentionNorm == null) {
    errors.push('normalization.postAttentionNorm is required');
  }
  if (inf.normalization.preFeedforwardNorm == null) {
    errors.push('normalization.preFeedforwardNorm is required');
  }
  if (inf.normalization.postFeedforwardNorm == null) {
    errors.push('normalization.postFeedforwardNorm is required');
  }

  // FFN fields
  if (inf.ffn.activation == null) {
    errors.push('ffn.activation is required');
  }
  if (inf.ffn.gatedActivation == null) {
    errors.push('ffn.gatedActivation is required');
  }

  // RoPE fields - non-nullable required
  if (inf.rope.ropeTheta == null) {
    errors.push('rope.ropeTheta is required');
  }
  if (inf.rope.ropeScalingFactor == null) {
    errors.push('rope.ropeScalingFactor is required (use 1.0 for no scaling)');
  }
  // RoPE fields - nullable required (undefined = missing, null = disabled)
  if (inf.rope.ropeScalingType === undefined) {
    errors.push('rope.ropeScalingType must be explicitly set (null for no scaling, or scaling type string)');
  }
  if (inf.rope.ropeLocalTheta === undefined) {
    errors.push('rope.ropeLocalTheta must be explicitly set (null for no local theta, or number)');
  }

  // Output fields - non-nullable required
  if (inf.output.tieWordEmbeddings == null) {
    errors.push('output.tieWordEmbeddings is required');
  }
  if (inf.output.scaleEmbeddings == null) {
    errors.push('output.scaleEmbeddings is required');
  }
  if (inf.output.embeddingTranspose == null) {
    errors.push('output.embeddingTranspose is required');
  }
  // Output fields - nullable required (undefined = missing, null = disabled)
  if (inf.output.finalLogitSoftcapping === undefined) {
    errors.push('output.finalLogitSoftcapping must be explicitly set (null for no softcapping, or number)');
  }
  if (inf.output.embeddingVocabSize === undefined) {
    errors.push('output.embeddingVocabSize must be explicitly set (null to use architecture.vocabSize, or number)');
  }

  if (errors.length > 0) {
    throw new Error(
      `Manifest "${modelId}" has incomplete inference config. ` +
      `Missing required fields:\n  - ${errors.join('\n  - ')}\n` +
      `Re-convert the model using the latest converter.`
    );
  }
}

/**
 * Convert MergedConfig to ParsedModelConfig.
 *
 * This is the manifest-first path that uses manifest.inference as the source
 * of truth instead of detecting model family from architecture strings.
 *
 * @param {import('../../config/merge.js').MergedConfig} merged
 * @param {import('./config.js').ManifestWithInference} manifest
 * @returns {import('./config.js').ParsedModelConfig}
 */
export function toParsedConfigFromMerged(merged, manifest) {
  const rawConfig = /** @type {import('./config.js').RawConfig} */ (manifest.config ?? {});
  const config = rawConfig.text_config ?? rawConfig;
  const inf = merged.inference;

  // Validate required fields are present (fail fast on incomplete manifests)
  validateRequiredInferenceFields(inf, merged.modelId);

  // Get architecture dimensions
  /** @type {import('../../config/schema/index.js').ArchitectureSchema} */
  let arch;
  if (typeof manifest.architecture === 'string') {
    // Fallback: infer from config
    arch = {
      numLayers: config.num_hidden_layers ?? config.n_layer ?? config.blockCount ?? 0,
      hiddenSize: config.hidden_size ?? config.n_embd ?? config.embeddingLength ?? 0,
      intermediateSize: config.intermediate_size ?? config.n_inner ?? config.feedForwardLength ?? 0,
      numAttentionHeads: config.num_attention_heads ?? config.n_head ?? config.attentionHeadCount ?? 0,
      numKeyValueHeads: config.num_key_value_heads ?? config.attentionHeadCountKV ?? config.num_attention_heads ?? config.n_head ?? 0,
      headDim: config.head_dim ?? Math.floor((config.hidden_size ?? 0) / (config.num_attention_heads ?? 1)),
      vocabSize: config.vocab_size ?? 0,
      maxSeqLen: config.max_position_embeddings ?? config.contextLength ?? DEFAULT_MAX_POSITION_EMBEDDINGS,
      // Use manifest inference as source of truth for RoPE (not raw config)
      ropeTheta: inf.rope.ropeTheta,
      rmsNormEps: config.rms_norm_eps ?? DEFAULT_RMS_NORM_EPS,
    };
  } else {
    arch = /** @type {import('../../config/schema/index.js').ArchitectureSchema} */ (manifest.architecture);
  }

  // Compute layer types from layerPattern
  /** @type {string[] | null} */
  let layerTypes = null;
  if (inf.layerPattern) {
    const numLayers = arch.numLayers;
    const patternType = inf.layerPattern.type;

    // Fail fast if alternating pattern lacks required globalPattern
    if (patternType === 'alternating' && inf.layerPattern.globalPattern == null) {
      throw new Error(
        `Manifest "${merged.modelId}" has layerPattern.type='alternating' but globalPattern is missing. ` +
        `Re-convert the model to include layerPattern.globalPattern.`
      );
    }

    // Fail fast if every_n pattern lacks required period
    if (patternType === 'every_n' && inf.layerPattern.period == null) {
      throw new Error(
        `Manifest "${merged.modelId}" has layerPattern.type='every_n' but period is missing. ` +
        `Re-convert the model to include layerPattern.period.`
      );
    }
    const period = inf.layerPattern.period ?? 1;  // Fallback only for non-every_n types

    if (patternType === 'alternating') {
      const pattern = inf.layerPattern.globalPattern;
      if (pattern === 'even') {
        layerTypes = Array.from({ length: numLayers }, (_, i) =>
          i % 2 === 0 ? 'full_attention' : 'sliding_attention'
        );
      } else if (pattern === 'odd') {
        layerTypes = Array.from({ length: numLayers }, (_, i) =>
          i % 2 === 1 ? 'full_attention' : 'sliding_attention'
        );
      }
    } else if (patternType === 'every_n') {
      layerTypes = Array.from({ length: numLayers }, (_, i) =>
        i % period === 0 ? 'full_attention' : 'sliding_attention'
      );
    }
  }

  // Compute queryPreAttnScalar from manifest inference (NOT from preset detection)
  // Manifest-first: queryPreAttnScalar is required in ManifestAttentionSchema
  const headDim = arch.headDim;
  const queryPreAttnScalar = inf.attention.queryPreAttnScalar;

  // Get stop token IDs (cast to Manifest for compatibility)
  const stopTokenIds = getStopTokenIds(config, /** @type {import('./config.js').Manifest} */ (manifest));

  // Get MoE config
  const useMoE = (config.num_local_experts ?? 0) > 1 || (config.num_experts ?? 0) > 1;
  const numExperts = config.num_local_experts ?? config.num_experts ?? 8;
  const moeTopK = config.experts_per_token ?? config.num_experts_per_tok ?? config.top_k ?? 2;

  // RoPE scaling - use manifest inference as source of truth (not raw config)
  const ropeScale = inf.rope.ropeScalingFactor;
  /** @type {string | null} */
  const ropeScalingType = inf.rope.ropeScalingType;
  // Build ropeScaling object from manifest values if scaling is enabled
  // Include YARN params when present
  /** @type {import('./config.js').RopeScalingConfig | null} */
  const ropeScaling = ropeScalingType ? {
    type: ropeScalingType,
    factor: ropeScale,
    ...(ropeScalingType === 'yarn' && inf.rope.yarnBetaFast != null && { beta_fast: inf.rope.yarnBetaFast }),
    ...(ropeScalingType === 'yarn' && inf.rope.yarnBetaSlow != null && { beta_slow: inf.rope.yarnBetaSlow }),
    ...(ropeScalingType === 'yarn' && inf.rope.yarnOriginalMaxPos != null && {
      original_max_position_embeddings: inf.rope.yarnOriginalMaxPos
    }),
  } : null;

  // Activation type
  const activation = inf.ffn.activation;
  /** @type {import('./config.js').ActivationType} */
  const hiddenActivation =
    activation === 'silu' || activation === 'swiglu' ? 'silu' :
    activation === 'gelu' || activation === 'geglu' ? 'gelu' : 'silu';

  const chatTemplateType = inf.chatTemplate?.type ?? null;
  const chatTemplateEnabled = inf.chatTemplate?.enabled ?? false;

  return {
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: arch.intermediateSize,
    numHeads: arch.numAttentionHeads,
    numKVHeads: arch.numKeyValueHeads,
    headDim: arch.headDim,
    vocabSize: arch.vocabSize,
    maxSeqLen: arch.maxSeqLen,
    useMoE,
    numExperts,
    moeTopK,
    slidingWindow: inf.attention.slidingWindow ?? null,
    ropeTheta: inf.rope.ropeTheta,
    ropeLocalTheta: inf.rope.ropeLocalTheta ?? null,
    ropeScale,
    ropeScalingType,
    ropeScaling,
    quantization: /** @type {string} */ (manifest.quantization) ?? 'f16',
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps: arch.rmsNormEps ?? 1e-5,
    rmsNormWeightOffset: inf.normalization.rmsNormWeightOffset,
    scaleEmbeddings: inf.output.scaleEmbeddings,
    useTiedEmbeddings: inf.output.tieWordEmbeddings,
    embeddingTranspose: inf.output.embeddingTranspose,
    embeddingVocabSize: inf.output.embeddingVocabSize,
    hiddenActivation,
    // Model detection flags - derived from manifest inference config values
    // Kept for backward compat until pipeline code reads config values directly
    isGemma3: inf.rope.ropeLocalTheta != null,  // Gemma 3 has local RoPE theta
    isGemma2: inf.attention.attnLogitSoftcapping != null,  // Gemma 2 has attn softcapping
    isLlama3Instruct: chatTemplateType === 'llama3',
    isQwen3: false,  // TODO: Add model family to manifest
    isGptOss: false,  // TODO: Add model family to manifest
    stopTokenIds,
    layerTypes,
    attentionBias: config.attention_bias ?? false,
    finalLogitSoftcapping: inf.output.finalLogitSoftcapping ?? null,
    attnLogitSoftcapping: inf.attention.attnLogitSoftcapping ?? null,
    queryKeyNorm: inf.attention.queryKeyNorm,
    queryPreAttnScalar,
    layerPipeline: null,  // TODO: Add to ManifestInferenceSchema if needed
    chatTemplateType,
    chatTemplateEnabled,
    kernelPath: inf.defaultKernelPath,
  };
}

/**
 * Parse model config from manifest using manifest-first resolution.
 *
 * This is the new entry point that uses manifest.inference as the source
 * of truth. It:
 * 1. Validates manifest has inference config
 * 2. Calls mergeConfig() to merge with runtime overrides
 * 3. Converts to ParsedModelConfig
 *
 * @param {import('./config.js').ManifestWithInference} manifest
 * @param {import('../../config/merge.js').RuntimeInferenceOverrides} [runtimeOverrides]
 * @returns {import('./config.js').ParsedModelConfig}
 */
export function parseModelConfigFromManifest(manifest, runtimeOverrides) {
  // Merge manifest inference with runtime overrides
  const merged = mergeConfig(
    {
      modelId: manifest.modelId ?? manifest.model_id ?? 'unknown',
      inference: manifest.inference,
      architecture: /** @type {import('../../config/schema/index.js').ArchitectureSchema | string | undefined} */ (manifest.architecture),
    },
    runtimeOverrides
  );

  // Log config source info
  const runtimeSources = Array.from(merged._sources.entries())
    .filter(([, src]) => src === 'runtime')
    .length;
  const totalSources = merged._sources.size;
  if (runtimeSources > 0) {
    log.info('Config', `Manifest-first config: ${totalSources - runtimeSources} from manifest, ${runtimeSources} from runtime`);
  } else {
    log.debug('Config', `Manifest-first config: ${totalSources} values from manifest`);
  }

  // Convert to ParsedModelConfig
  return toParsedConfigFromMerged(merged, manifest);
}

// =============================================================================
// Main Entry Point
// =============================================================================

/**
 * Parse model configuration from manifest.
 *
 * Requires manifest.inference to be present (manifest-first architecture).
 * Legacy manifests without inference config must be re-converted.
 *
 * @param {import('./config.js').Manifest} manifest
 * @param {import('../../config/merge.js').RuntimeInferenceOverrides} [runtimeOverrides]
 * @returns {import('./config.js').ParsedModelConfig}
 * @throws Error if manifest.inference is missing
 */
export function parseModelConfig(manifest, runtimeOverrides) {
  // Manifest-first architecture: inference config is required
  if (!hasManifestInference(manifest)) {
    const modelId = manifest.modelId ?? manifest.model_id ?? 'unknown';
    throw new Error(
      `Manifest "${modelId}" is missing inference config. ` +
      `Re-convert the model using the latest converter to add manifest.inference. ` +
      `Legacy preset-based resolution has been removed.`
    );
  }

  log.info('Config', 'Using manifest-first config (source of truth)');
  return parseModelConfigFromManifest(manifest, runtimeOverrides);
}
