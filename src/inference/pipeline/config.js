import { log } from '../../debug/index.js';
import { mergeConfig } from '../../config/merge.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

// =============================================================================
// Model Detection Functions
// =============================================================================

export function getStopTokenIds(config, manifest) {
  // Priority: manifest.eos_token_id > config.eos_token_id > config.text_config.eos_token_id
  // Model-specific fallbacks are NOT allowed - converter must set eos_token_id
  const eosTokenId = manifest?.eos_token_id ?? config?.eos_token_id ?? config?.text_config?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  return [];
}

// =============================================================================
// Manifest-First Config Resolution (NEW)
// =============================================================================


export function hasManifestInference(manifest) {
  return 'inference' in manifest && manifest.inference != null;
}


function validateRequiredInferenceFields(inf, modelId) {
  
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
  if (inf.normalization.rmsNormEps == null) {
    errors.push('normalization.rmsNormEps is required');
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
  if (inf.ffn.swigluLimit !== undefined) {
    const limit = inf.ffn.swigluLimit;
    if (limit !== null && (typeof limit !== 'number' || Number.isNaN(limit) || limit <= 0)) {
      errors.push('ffn.swigluLimit must be a positive number or null');
    }
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

  // Layer pattern fields
  if (inf.layerPattern?.type == null) {
    errors.push('layerPattern.type is required');
  }
  if (inf.layerPattern?.globalPattern === undefined) {
    errors.push('layerPattern.globalPattern must be explicitly set (null if not applicable)');
  }
  if (inf.layerPattern?.period === undefined) {
    errors.push('layerPattern.period must be explicitly set (null if not applicable)');
  }

  // Chat template fields
  if (inf.chatTemplate?.type === undefined) {
    errors.push('chatTemplate.type must be explicitly set (null for no template)');
  }
  if (inf.chatTemplate?.enabled == null) {
    errors.push('chatTemplate.enabled is required');
  }

  // RoPE YARN fields
  if (inf.rope.yarnBetaFast === undefined) {
    errors.push('rope.yarnBetaFast must be explicitly set (null if not YARN)');
  }
  if (inf.rope.yarnBetaSlow === undefined) {
    errors.push('rope.yarnBetaSlow must be explicitly set (null if not YARN)');
  }
  if (inf.rope.yarnOriginalMaxPos === undefined) {
    errors.push('rope.yarnOriginalMaxPos must be explicitly set (null if not YARN)');
  }

  if (errors.length > 0) {
    throw new Error(
      `Manifest "${modelId}" has incomplete inference config. ` +
      `Missing required fields:\n  - ${errors.join('\n  - ')}\n` +
      `Re-convert the model using the latest converter.`
    );
  }
}


export function toParsedConfigFromMerged(merged, manifest) {
  const rawConfig = manifest.config ?? {};
  const config = rawConfig.text_config ?? rawConfig;
  const inf = merged.inference;

  // Validate required fields are present (fail fast on incomplete manifests)
  validateRequiredInferenceFields(inf, merged.modelId);
  if (manifest.quantization == null) {
    throw new Error(`Manifest "${merged.modelId}" is missing quantization.`);
  }

  // Get architecture dimensions
  const arch = (manifest.architecture && typeof manifest.architecture === 'object')
    ? manifest.architecture
    : null;
  if (!arch) {
    throw new Error(
      `Manifest "${merged.modelId}" is missing architecture config. ` +
      `Re-convert the model using the latest converter to add manifest.architecture.`
    );
  }

  // Compute layer types from layerPattern
  
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
    const period = inf.layerPattern.period;
    const pattern = inf.layerPattern.globalPattern;
    const patternKind = selectRuleValue(
      'inference',
      'layerPattern',
      'patternKind',
      { patternType, globalPattern: pattern }
    );
    if (patternKind) {
      layerTypes = Array.from({ length: numLayers }, (_, i) => {
        const isEven = i % 2 === 0;
        const isStride = period == null ? false : i % period === 0;
        return selectRuleValue(
          'inference',
          'layerPattern',
          'layerType',
          { patternKind, isEven, isStride }
        );
      });
    }
  }

  // Compute queryPreAttnScalar from manifest inference (NOT from preset detection)
  // Manifest-first: queryPreAttnScalar is required in ManifestAttentionSchema
  const headDim = arch.headDim;
  const queryPreAttnScalar = inf.attention.queryPreAttnScalar;

  // Get stop token IDs (cast to Manifest for compatibility)
  const stopTokenIds = getStopTokenIds(config, manifest);

  // Get MoE config
  const moeConfig = manifest.moeConfig ?? null;
  const useMoE = (moeConfig?.numExperts ?? 0) > 1;
  if (useMoE && (moeConfig?.numExperts == null || moeConfig?.numExpertsPerToken == null)) {
    throw new Error(`Manifest "${manifest.modelId}" is missing moeConfig fields for MoE inference.`);
  }
  const numExperts = useMoE ? moeConfig.numExperts : 0;
  const moeTopK = useMoE ? moeConfig.numExpertsPerToken : 0;

  // RoPE scaling - use manifest inference as source of truth (not raw config)
  const ropeScale = inf.rope.ropeScalingFactor;
  const ropeScalingType = inf.rope.ropeScalingType;
  // Build ropeScaling object from manifest values if scaling is enabled
  // Include YARN params when present
  
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
  
  const hiddenActivation = selectRuleValue(
    'inference',
    'config',
    'hiddenActivation',
    { activation }
  );

  const chatTemplateType = inf.chatTemplate.type;
  const chatTemplateEnabled = inf.chatTemplate.enabled;

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
    slidingWindow: inf.attention.slidingWindow,
    ropeTheta: inf.rope.ropeTheta,
    ropeLocalTheta: inf.rope.ropeLocalTheta,
    ropeScale,
    ropeScalingType,
    ropeScaling,
    quantization: manifest.quantization,
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps: inf.normalization.rmsNormEps,
    rmsNormWeightOffset: inf.normalization.rmsNormWeightOffset,
    scaleEmbeddings: inf.output.scaleEmbeddings,
    useTiedEmbeddings: inf.output.tieWordEmbeddings,
    embeddingTranspose: inf.output.embeddingTranspose,
    embeddingVocabSize: inf.output.embeddingVocabSize,
    hiddenActivation,
    swigluLimit: inf.ffn.swigluLimit,
    // Model detection flags - derived from manifest inference config values
    // Kept for backward compat until pipeline code reads config values directly
    isGemma3: inf.rope.ropeLocalTheta != null,  // Gemma 3 has local RoPE theta
    isGemma2: inf.attention.attnLogitSoftcapping != null,  // Gemma 2 has attn softcapping
    isLlama3Instruct: chatTemplateType === 'llama3',
    stopTokenIds,
    layerTypes,
    attentionBias: config.attention_bias ?? false,
    finalLogitSoftcapping: inf.output.finalLogitSoftcapping,
    attnLogitSoftcapping: inf.attention.attnLogitSoftcapping,
    queryKeyNorm: inf.attention.queryKeyNorm,
    queryPreAttnScalar,
    layerPipeline: inf.pipeline ?? null,
    chatTemplateType,
    chatTemplateEnabled,
    kernelPath: inf.defaultKernelPath,
  };
}


export function parseModelConfigFromManifest(manifest, runtimeOverrides) {
  // Merge manifest inference with runtime overrides
  const merged = mergeConfig(
    {
      modelId: manifest.modelId ?? manifest.model_id ?? 'unknown',
      inference: manifest.inference,
      architecture: manifest.architecture,
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
