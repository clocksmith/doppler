import { log } from '../../../debug/index.js';
import { mergeConfig } from '../../../config/merge.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

const UNSUPPORTED_RUNTIME_MODEL_TYPES = new Set(['mamba', 'rwkv']);

// =============================================================================
// Model Detection Functions
// =============================================================================

function assertSupportedRuntimeModelType(manifest) {
  const modelType = typeof manifest?.modelType === 'string'
    ? manifest.modelType.trim().toLowerCase()
    : '';
  if (!modelType) return;
  if (!UNSUPPORTED_RUNTIME_MODEL_TYPES.has(modelType)) return;

  const modelId = manifest?.modelId ?? 'unknown';
  throw new Error(
    `Manifest "${modelId}" declares modelType "${modelType}", but that runtime family is not implemented yet.`
  );
}

function resolveRotaryDim(headDim, partialRotaryFactor, modelId) {
  if (partialRotaryFactor == null) {
    return headDim;
  }
  if (typeof partialRotaryFactor !== 'number' || Number.isNaN(partialRotaryFactor)) {
    throw new Error(`Manifest "${modelId}" has invalid rope.partialRotaryFactor.`);
  }
  if (partialRotaryFactor <= 0 || partialRotaryFactor > 1) {
    throw new Error(
      `Manifest "${modelId}" requires 0 < rope.partialRotaryFactor <= 1; got ${partialRotaryFactor}.`
    );
  }
  const rotaryDim = Math.trunc(headDim * partialRotaryFactor);
  if (rotaryDim <= 0 || (rotaryDim % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" resolves rope rotary dim ${rotaryDim} from headDim=${headDim} ` +
      `and partialRotaryFactor=${partialRotaryFactor}, but rotary dim must be a positive even integer.`
    );
  }
  return rotaryDim;
}

export function getStopTokenIds(manifest) {
  const eosTokenId = manifest?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  const modelId = manifest?.modelId ?? 'unknown';
  throw new Error(
    `Manifest "${modelId}" is missing eos_token_id. Re-convert the model with tokenizer metadata.`
  );
}

function normalizeFfnTensorShape(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
  if (rows <= 0 || cols <= 0) return null;
  return [Math.trunc(rows), Math.trunc(cols)];
}

function isExpertTensorName(name) {
  const lower = String(name || '').toLowerCase();
  return lower.includes('.experts.') || lower.includes('.expert.') || lower.includes('block_sparse_moe');
}

function inferLfm2IntermediateSizeFromManifest(manifest) {
  const tensors = manifest?.tensors;
  if (!tensors || typeof tensors !== 'object') return null;
  const candidates = [];
  for (const [name, entry] of Object.entries(tensors)) {
    if (!name || isExpertTensorName(name)) continue;
    const shape = normalizeFfnTensorShape(entry?.shape);
    if (!shape) continue;
    const lower = name.toLowerCase();
    if (
      lower.endsWith('.feed_forward.w1.weight')
      || lower.endsWith('.feed_forward.w3.weight')
      || lower.endsWith('.ffn_gate.weight')
      || lower.endsWith('.ffn_up.weight')
      || lower.endsWith('.ffn.gate_proj.weight')
      || lower.endsWith('.ffn.up_proj.weight')
      || lower.endsWith('.mlp.gate_proj.weight')
      || lower.endsWith('.mlp.up_proj.weight')
    ) {
      candidates.push(shape[0]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w2.weight')
      || lower.endsWith('.ffn_down.weight')
      || lower.endsWith('.ffn.down_proj.weight')
      || lower.endsWith('.mlp.down_proj.weight')
    ) {
      candidates.push(shape[1]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w1_w3.weight')
      || lower.endsWith('.ffn_gate_up.weight')
      || lower.endsWith('.ffn.gate_up_proj.weight')
      || lower.endsWith('.mlp.gate_up_proj.weight')
    ) {
      if (shape[0] % 2 === 0) {
        candidates.push(Math.trunc(shape[0] / 2));
      }
    }
  }
  if (candidates.length === 0) return null;
  const counts = new Map();
  for (const value of candidates) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0] - b[0];
    })[0]?.[0] ?? null;
}

function resolveIntermediateSizeForRuntime(manifest, inf, arch, modelId) {
  const fromArch = arch?.intermediateSize;
  if (typeof fromArch !== 'number' || !Number.isFinite(fromArch) || fromArch <= 0) {
    return fromArch;
  }
  const presetId = String(manifest?.inference?.presetId ?? inf?.presetId ?? '').toLowerCase();
  if (presetId !== 'lfm2') {
    return fromArch;
  }
  const inferred = inferLfm2IntermediateSizeFromManifest(manifest);
  if (inferred == null || inferred === fromArch) {
    return fromArch;
  }
  throw new Error(
    `Manifest "${modelId}" has intermediateSize=${fromArch}, but FFN tensors imply ${inferred}. ` +
    'Re-convert the model so manifest architecture matches the weights.'
  );
}

// =============================================================================
// Manifest-First Config Resolution (NEW)
// =============================================================================


export function hasManifestInference(manifest) {
  return 'inference' in manifest && manifest.inference != null;
}


export function validateRequiredInferenceFields(inf, modelId) {
  inf = inf ?? {};
  inf.attention = inf.attention ?? {};
  inf.normalization = inf.normalization ?? {};
  inf.ffn = inf.ffn ?? {};
  inf.rope = inf.rope ?? {};
  inf.output = inf.output ?? {};
  inf.layerPattern = inf.layerPattern ?? {};
  inf.chatTemplate = inf.chatTemplate ?? {};
  const errors = [];

  // Attention fields - non-nullable required
  if (inf.attention.queryPreAttnScalar == null) {
    errors.push('attention.queryPreAttnScalar is required');
  }
  if (inf.attention.queryKeyNorm == null) {
    errors.push('attention.queryKeyNorm is required');
  }
  if (inf.attention.attentionBias == null) {
    errors.push('attention.attentionBias is required');
  }
  if (inf.attention.causal == null) {
    errors.push('attention.causal is required');
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
  if (inf.ffn.swigluLimit === undefined) {
    errors.push('ffn.swigluLimit must be explicitly set (null for no limit, or number)');
  } else {
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
  if (inf.rope.mropeInterleaved == null) {
    errors.push('rope.mropeInterleaved is required');
  }
  if (inf.rope.mropeSection === undefined) {
    errors.push('rope.mropeSection must be explicitly set (null when unused, or an array of positive integers)');
  }
  if (inf.rope.partialRotaryFactor === undefined) {
    errors.push('rope.partialRotaryFactor must be explicitly set (null when unused, or a number in (0, 1])');
  } else {
    const factor = inf.rope.partialRotaryFactor;
    if (factor !== null && (typeof factor !== 'number' || Number.isNaN(factor) || factor <= 0 || factor > 1)) {
      errors.push('rope.partialRotaryFactor must be a number in (0, 1] or null');
    }
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
  if (inf.layerPattern?.offset === undefined) {
    errors.push('layerPattern.offset must be explicitly set (null if not applicable)');
  }
  if (inf.layerPattern?.type === 'custom' && inf.layerPattern?.layerTypes === undefined) {
    errors.push('layerPattern.layerTypes must be explicitly set for custom patterns');
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

function normalizeLayerTypeTag(value) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
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
    || normalized === 'local'
    || normalized === 'sliding'
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
  if (normalized === 'moe' || normalized === 'mamba' || normalized === 'rwkv') {
    return normalized;
  }
  return null;
}

function parseCustomLayerTypes(layerTypes, numLayers, modelId) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
    throw new Error(
      `Manifest "${modelId}" has layerPattern.type='custom' but layerPattern.layerTypes is missing or empty. ` +
      'Re-convert the model to include explicit layer types.'
    );
  }
  if (layerTypes.length !== numLayers) {
    throw new Error(
      `Manifest "${modelId}" has layerPattern.type='custom' with ${layerTypes.length} layer types, ` +
      `expected ${numLayers}. Re-convert the model to preserve full per-layer metadata.`
    );
  }
  return layerTypes.map((layerType, index) => {
    const normalized = normalizeLayerTypeTag(layerType);
    if (!normalized) {
      throw new Error(
        `Manifest "${modelId}" has unknown layerPattern.layerTypes[${index}]="${layerType}". ` +
        'Supported types: conv, full_attention, sliding_attention, linear_attention, moe, mamba, rwkv.'
      );
    }
    return normalized;
  });
}

function parseLinearNormMode(value, sharedFlag = null, modelId = 'unknown') {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'shared') return 'shared';
    if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
      return 'per_head';
    }
    throw new Error(
      `Manifest "${modelId}" has unsupported linear_norm_mode="${value}". ` +
      'Supported values: "shared", "per_head".'
    );
  }
  if (typeof sharedFlag === 'boolean') {
    return sharedFlag ? 'shared' : 'per_head';
  }
  return null;
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
  const resolvedIntermediateSize = resolveIntermediateSizeForRuntime(manifest, inf, arch, merged.modelId);

  // Compute layer types from layerPattern
  
  let layerTypes = null;
  if (inf.layerPattern) {
    const numLayers = arch.numLayers;
    const patternType = inf.layerPattern.type;

    if (patternType === 'custom') {
      layerTypes = parseCustomLayerTypes(inf.layerPattern.layerTypes, numLayers, merged.modelId);
    } else {
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
      const rawOffset = inf.layerPattern.offset;
      const offset = (
        Number.isFinite(rawOffset) && period != null && period > 0
      )
        ? ((Math.trunc(rawOffset) % period) + period) % period
        : 0;
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
          // For every_n pattern: global at layer "offset" and every N thereafter.
          // e.g. period=6, offset=5 => indices 5,11,17,...
          const isStride = period == null ? false : (((i - offset) % period + period) % period) === 0;
          return selectRuleValue(
            'inference',
            'layerPattern',
            'layerType',
            { patternKind, isEven, isStride }
          );
        });
      }
    }
  }

  if (!Array.isArray(layerTypes) && Array.isArray(config.layer_types) && config.layer_types.length > 0) {
    layerTypes = parseCustomLayerTypes(config.layer_types, arch.numLayers, merged.modelId);
  }

  // Compute queryPreAttnScalar from manifest inference (NOT from preset detection)
  // Manifest-first: queryPreAttnScalar is required in ManifestAttentionSchema
  const headDim = arch.headDim;
  const queryPreAttnScalar = inf.attention.queryPreAttnScalar;
  const causalAttention = inf.attention.causal;

  // Cross-field sanity: queryPreAttnScalar should typically equal headDim.
  // A value of sqrt(headDim) indicates a known converter bug that produces
  // attnScale = 1/sqrt(sqrt(headDim)) instead of the correct 1/sqrt(headDim).
  if (queryPreAttnScalar != null && headDim != null
      && queryPreAttnScalar !== headDim
      && Math.abs(queryPreAttnScalar - Math.sqrt(headDim)) < 0.01) {
    throw new Error(
      `Model "${merged.modelId}": queryPreAttnScalar (${queryPreAttnScalar}) ` +
      `equals sqrt(headDim) instead of headDim (${headDim}). ` +
      `This is a known converter bug — the manifest must be regenerated ` +
      `with the corrected converter.`
    );
  }

  // Get stop token IDs (cast to Manifest for compatibility)
  const stopTokenIds = getStopTokenIds(manifest);

  // Get MoE config
  const moeConfig = manifest.moeConfig ?? null;
  const useMoE = (moeConfig?.numExperts ?? 0) > 1;
  if (useMoE && (moeConfig?.numExperts == null || moeConfig?.numExpertsPerToken == null || !moeConfig?.expertFormat)) {
    throw new Error(`Manifest "${manifest.modelId}" is missing moeConfig fields for MoE inference.`);
  }
  const numExperts = useMoE ? moeConfig.numExperts : 0;
  const moeTopK = useMoE ? moeConfig.numExpertsPerToken : 0;
  const expertFormat = useMoE ? moeConfig.expertFormat : null;

  // RoPE scaling - use manifest inference as source of truth (not raw config)
  const ropeScale = inf.rope.ropeScalingFactor;
  const ropeScalingType = inf.rope.ropeScalingType;
  const ropeLocalScale = inf.rope.ropeLocalScalingFactor ?? ropeScale;
  const ropeLocalScalingType = inf.rope.ropeLocalScalingType ?? ropeScalingType;
  const partialRotaryFactor = inf.rope.partialRotaryFactor;
  const ropeInterleaved = inf.rope.mropeInterleaved === true;
  const mropeSection = Array.isArray(inf.rope.mropeSection)
    ? inf.rope.mropeSection.map((entry) => Math.trunc(Number(entry)))
    : null;
  const ropeRotaryDim = resolveRotaryDim(arch.headDim, partialRotaryFactor, merged.modelId);
  if (mropeSection && mropeSection.some((entry) => !Number.isFinite(entry) || entry <= 0)) {
    throw new Error(
      `Manifest "${merged.modelId}" has invalid rope.mropeSection; expected positive integers.`
    );
  }
  if (ropeInterleaved && mropeSection) {
    const doubledMropeDim = mropeSection.reduce((sum, entry) => sum + entry, 0) * 2;
    if (doubledMropeDim !== ropeRotaryDim) {
      throw new Error(
        `Manifest "${merged.modelId}" declares rope.mropeSection=${JSON.stringify(mropeSection)}, ` +
        `which expands to rotary dim ${doubledMropeDim}, but the resolved rotary dim is ${ropeRotaryDim}.`
      );
    }
  }

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
  const ropeLocalScaling = ropeLocalScalingType ? {
    type: ropeLocalScalingType,
    factor: ropeLocalScale,
    ...(ropeLocalScalingType === 'yarn' && (inf.rope.ropeLocalYarnBetaFast ?? inf.rope.yarnBetaFast) != null && {
      beta_fast: inf.rope.ropeLocalYarnBetaFast ?? inf.rope.yarnBetaFast
    }),
    ...(ropeLocalScalingType === 'yarn' && (inf.rope.ropeLocalYarnBetaSlow ?? inf.rope.yarnBetaSlow) != null && {
      beta_slow: inf.rope.ropeLocalYarnBetaSlow ?? inf.rope.yarnBetaSlow
    }),
    ...(ropeLocalScalingType === 'yarn'
      && (inf.rope.ropeLocalYarnOriginalMaxPos ?? inf.rope.yarnOriginalMaxPos) != null && {
      original_max_position_embeddings:
        inf.rope.ropeLocalYarnOriginalMaxPos ?? inf.rope.yarnOriginalMaxPos
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
  const parsePositiveInt = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return null;
    return Math.trunc(num);
  };

  const linearNumKeyHeads = parsePositiveInt(arch.linearNumKeyHeads ?? config.linear_num_key_heads);
  const linearNumValueHeads = parsePositiveInt(arch.linearNumValueHeads ?? config.linear_num_value_heads);
  const linearKeyHeadDim = parsePositiveInt(arch.linearKeyHeadDim ?? config.linear_key_head_dim);
  const linearValueHeadDim = parsePositiveInt(arch.linearValueHeadDim ?? config.linear_value_head_dim);
  const linearConvKernelDim = parsePositiveInt(arch.linearConvKernelDim ?? config.linear_conv_kernel_dim);
  const linearNormMode = parseLinearNormMode(
    arch.linearNormMode ?? config.linear_norm_mode,
    config.linear_norm_shared,
    merged.modelId
  );

  return {
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: resolvedIntermediateSize,
    numHeads: arch.numAttentionHeads,
    numKVHeads: arch.numKeyValueHeads,
    headDim: arch.headDim,
    vocabSize: arch.vocabSize,
    maxSeqLen: arch.maxSeqLen,
    useMoE,
    numExperts,
    moeTopK,
    expertFormat,
    slidingWindow: inf.attention.slidingWindow,
    ropeTheta: inf.rope.ropeTheta,
    ropeLocalTheta: inf.rope.ropeLocalTheta,
    ropeRotaryDim,
    ropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeScale,
    ropeLocalScale,
    ropeScalingType,
    ropeLocalScalingType,
    ropeScaling,
    ropeLocalScaling,
    quantization: manifest.quantization,
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps: inf.normalization.rmsNormEps,
    rmsNormWeightOffset: inf.normalization.rmsNormWeightOffset,
    postAttentionNorm: inf.normalization.postAttentionNorm,
    preFeedforwardNorm: inf.normalization.preFeedforwardNorm,
    postFeedforwardNorm: inf.normalization.postFeedforwardNorm,
    scaleEmbeddings: inf.output.scaleEmbeddings,
    useTiedEmbeddings: inf.output.tieWordEmbeddings,
    embeddingTranspose: inf.output.embeddingTranspose,
    embeddingVocabSize: inf.output.embeddingVocabSize,
    hiddenActivation,
    swigluLimit: inf.ffn.swigluLimit,
    stopTokenIds,
    layerTypes,
    linearNumKeyHeads,
    linearNumValueHeads,
    linearKeyHeadDim,
    linearValueHeadDim,
    linearConvKernelDim,
    linearNormMode,
    attentionBias: inf.attention.attentionBias,
    causalAttention,
    finalLogitSoftcapping: inf.output.finalLogitSoftcapping,
    attnLogitSoftcapping: inf.attention.attnLogitSoftcapping,
    queryKeyNorm: inf.attention.queryKeyNorm,
    attentionOutputGate: inf.attention.attentionOutputGate === true,
    queryPreAttnScalar,
    layerPipeline: inf.pipeline ?? null,
    chatTemplateType,
    chatTemplateEnabled,
    kernelPath: inf.defaultKernelPath,
  };
}


export function parseModelConfigFromManifest(manifest, runtimeOverrides) {
  assertSupportedRuntimeModelType(manifest);

  // Merge manifest inference with runtime overrides
  const merged = mergeConfig(
    {
      modelId: manifest.modelId ?? 'unknown',
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
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(
      `Manifest "${modelId}" is missing inference config. ` +
      `Re-convert the model using the latest converter to add manifest.inference. ` +
      `Legacy preset-based resolution has been removed.`
    );
  }

  log.info('Config', 'Using manifest-first config (source of truth)');
  return parseModelConfigFromManifest(manifest, runtimeOverrides);
}
