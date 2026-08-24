import { log } from '../../../debug/index.js';
import { mergeConfig, dumpConfigSources } from '../../../config/merge.js';
import { validateModelOverrides } from '../../../config/param-validator.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { appendHeterogeneousAttentionValidation, resolveHeterogeneousAttentionContract } from './attention/heterogeneous-contract.js';
import { resolvePostNormContract } from './normalization-contract.js';
import { resolveEmbeddingNormalization } from './embedding-contract.js';
import {
  PER_LAYER_INPUT_MATERIALIZATION_MODES,
  PER_LAYER_INPUT_ROW_CACHE_MODES,
  PER_LAYER_INPUT_PREFETCH_MODES,
  PER_LAYER_INPUT_GPU_UPLOAD_MODES,
  PER_LAYER_INPUT_HOT_CACHE_MODES,
  PREFILL_CHUNK_SUBMIT_MODES,
} from '../../../config/schema/execution-v1.schema.js';
import { assertSupportedManifestInference, validateLayerIntermediateSizesAgainstManifest, validateRequiredInferenceFields } from './config/validation.js';
import { resolveAudioConfig, resolvePerLayerInputsSession, resolveSessionSettings, resolveVisionConfig } from './config/normalization.js';
export { assertSupportedManifestInference, validateRequiredInferenceFields } from './config/validation.js';

const UNSUPPORTED_RUNTIME_MODEL_TYPES = new Set(['mamba', 'rwkv']);

const KNOWN_CHAT_TEMPLATE_TYPES = new Set([
  'gemma',
  'gemma4',
  'llama3',
  'gpt-oss',
  'chatml',
  'qwen',
  'translategemma',
]);

function validateChatTemplateType(type, modelId) {
  if (type === null || type === undefined) return true;
  if (KNOWN_CHAT_TEMPLATE_TYPES.has(type)) return true;
  throw new Error(
    `Manifest "${modelId}" declares chatTemplate.type="${type}" which is not a known formatter type. ` +
    `Known types: ${[...KNOWN_CHAT_TEMPLATE_TYPES].join(', ')}. Re-convert the model or fix the manifest.`
  );
}

function resolveLargeWeightsConfig(inferenceConfig, modelId) {
  const lw = inferenceConfig?.largeWeights;
  if (lw === undefined || lw === null) return { gpuResidentOverrides: null };
  if (typeof lw !== 'object' || Array.isArray(lw)) {
    throw new Error(`Manifest "${modelId}" has invalid inference.largeWeights (must be object).`);
  }
  const overrides = lw.gpuResidentOverrides;
  if (overrides !== null && overrides !== undefined) {
    if (!Array.isArray(overrides) || !overrides.every((v) => typeof v === 'string' && v.length > 0)) {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.largeWeights.gpuResidentOverrides; ` +
        `expected array of non-empty strings or null.`
      );
    }
  }
  return { gpuResidentOverrides: overrides ?? null };
}

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

function resolveFrequencyBaseDim(headDim, rotaryDim, frequencyBaseDim, modelId, fieldName) {
  if (frequencyBaseDim == null) {
    return rotaryDim;
  }
  if (typeof frequencyBaseDim !== 'number' || Number.isNaN(frequencyBaseDim)) {
    throw new Error(`Manifest "${modelId}" has invalid ${fieldName}.`);
  }
  const resolved = Math.trunc(frequencyBaseDim);
  if (resolved <= 0 || (resolved % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} to be a positive even integer; got ${frequencyBaseDim}.`
    );
  }
  if (resolved < rotaryDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be >= rotary dim (${rotaryDim}).`
    );
  }
  if (resolved > headDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be <= attention head dim (${headDim}).`
    );
  }
  return resolved;
}

export function getStopTokenIds(manifest) {
  const eosTokenId = manifest?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  const modelId = manifest?.modelId ?? 'unknown';
  if (eosTokenId == null) {
    if (manifest?.modelType === 'embedding') {
      return [];
    }
    throw new Error(
      `Manifest "${modelId}" is missing eos_token_id. Re-convert the model with tokenizer metadata.`
    );
  }
  throw new Error(
    `Manifest "${modelId}" has eos_token_id of unsupported type "${typeof eosTokenId}" (value: ${JSON.stringify(eosTokenId)}). ` +
    'Expected a number or array of numbers. Re-convert the model with tokenizer metadata.'
  );
}

function resolveIntermediateSizeForRuntime(arch) {
  return arch?.intermediateSize;
}

function buildPerLayerIntermediateSizes({
  baseIntermediateSize,
  numLayers,
  numKvSharedLayers,
  useDoubleWideMlp,
  modelId,
}) {
  if (!Number.isFinite(baseIntermediateSize) || baseIntermediateSize <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.intermediateSize (${String(baseIntermediateSize)}).`
    );
  }
  if (!Number.isFinite(numLayers) || numLayers <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.numLayers (${String(numLayers)}).`
    );
  }

  const resolvedBaseIntermediateSize = Math.trunc(baseIntermediateSize);
  const resolvedNumLayers = Math.trunc(numLayers);
  const intermediateSizes = new Array(resolvedNumLayers).fill(resolvedBaseIntermediateSize);

  if (!useDoubleWideMlp) {
    return intermediateSizes;
  }

  if (!Number.isFinite(numKvSharedLayers) || numKvSharedLayers <= 0 || numKvSharedLayers >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but architecture.numKvSharedLayers=${String(numKvSharedLayers)} ` +
      `must be a positive integer smaller than numLayers=${resolvedNumLayers}.`
    );
  }

  const firstKvSharedLayerIdx = resolvedNumLayers - Math.trunc(numKvSharedLayers);
  if (firstKvSharedLayerIdx <= 0 || firstKvSharedLayerIdx >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but the derived first KV-shared layer index ` +
      `(${firstKvSharedLayerIdx}) is invalid for numLayers=${resolvedNumLayers}.`
    );
  }

  const widenedIntermediateSize = resolvedBaseIntermediateSize * 2;
  for (let layerIdx = firstKvSharedLayerIdx; layerIdx < resolvedNumLayers; layerIdx += 1) {
    intermediateSizes[layerIdx] = widenedIntermediateSize;
  }

  return intermediateSizes;
}

export function resolveLayerIntermediateSize(config, layerIdx) {
  const intermediateSizes = Array.isArray(config?.intermediateSizes) ? config.intermediateSizes : null;
  if (Number.isFinite(layerIdx) && intermediateSizes) {
    const resolved = intermediateSizes[Math.trunc(layerIdx)];
    if (Number.isFinite(resolved) && resolved > 0) {
      return Math.trunc(resolved);
    }
  }

  const fallback = Number(config?.intermediateSize);
  if (!Number.isFinite(fallback) || fallback <= 0) {
    throw new Error(`Invalid modelConfig.intermediateSize: ${String(config?.intermediateSize)}`);
  }
  return Math.trunc(fallback);
}

// =============================================================================
// Manifest-First Config Resolution (NEW)
// =============================================================================

function hasManifestInference(manifest) {
  return 'inference' in manifest && manifest.inference != null;
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

function toParsedConfigFromMerged(merged, manifest) {
  const mergedConfig = merged?.config ?? manifest.config ?? {};
  const rawConfig = mergedConfig.text_config ?? mergedConfig;
  const config = {
    ...rawConfig,
    ...(merged?.vision_config !== null && merged?.vision_config !== undefined
      ? { vision_config: merged.vision_config }
      : {}),
    ...(merged?.audio_config !== null && merged?.audio_config !== undefined
      ? { audio_config: merged.audio_config }
      : {}),
    ...(merged?.quantization_config !== null && merged?.quantization_config !== undefined
      ? { quantization_config: merged.quantization_config }
      : {}),
  };
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
  const resolvedIntermediateSize = resolveIntermediateSizeForRuntime(arch);
  const archNumHeads = Number(arch.numAttentionHeads ?? arch.numHeads);
  const archNumKVHeads = Number(arch.numKeyValueHeads ?? arch.numKVHeads);
  const archHeadDim = Number(arch.headDim);
  const archGlobalHeadDimRaw = arch.globalHeadDim ?? null;
  const archGlobalHeadDim = (
    typeof archGlobalHeadDimRaw === 'number' && Number.isFinite(archGlobalHeadDimRaw) && archGlobalHeadDimRaw > 0
  )
    ? Math.trunc(archGlobalHeadDimRaw)
    : null;
  const archNumKvSharedLayersRaw = arch.numKvSharedLayers ?? 0;
  const archNumKvSharedLayers = (
    typeof archNumKvSharedLayersRaw === 'number'
      && Number.isFinite(archNumKvSharedLayersRaw)
      && archNumKvSharedLayersRaw >= 0
  )
    ? Math.trunc(archNumKvSharedLayersRaw)
    : 0;
  const intermediateSizes = buildPerLayerIntermediateSizes({
    baseIntermediateSize: resolvedIntermediateSize,
    numLayers: arch.numLayers,
    numKvSharedLayers: archNumKvSharedLayers,
    useDoubleWideMlp: inf.ffn.useDoubleWideMlp,
    modelId: merged.modelId,
  });
  const maxIntermediateSize = Math.max(...intermediateSizes);
  validateLayerIntermediateSizesAgainstManifest(
    manifest,
    arch.hiddenSize,
    intermediateSizes,
    merged.modelId
  );

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

  // Compute queryPreAttnScalar from manifest inference (NOT from family detection)
  // Manifest-first: queryPreAttnScalar is required in ManifestAttentionSchema
  const headDim = archHeadDim;
  const queryPreAttnScalar = inf.attention.queryPreAttnScalar;
  const causalAttention = inf.attention.causal;

  // Preserve the manifest scalar exactly. Gemma-family models legitimately use
  // queryPreAttnScalar=1, but sqrt(headDim) is still a known converter bug that
  // produces attnScale = 1/sqrt(sqrt(headDim)) instead of the intended value.
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
  const embeddingPostprocessor = inf.output.embeddingPostprocessor;
  if (embeddingPostprocessor) {
    if (embeddingPostprocessor.includePrompt !== true) {
      throw new Error(
        `Manifest "${merged.modelId}" requires output.embeddingPostprocessor.includePrompt=false, ` +
        'but prompt-token masking is not implemented for embedding extraction.'
      );
    }
    let expectedInputSize = arch.hiddenSize;
    for (let i = 0; i < embeddingPostprocessor.projections.length; i++) {
      const projection = embeddingPostprocessor.projections[i];
      if (projection.inputSize !== expectedInputSize) {
        throw new Error(
          `Manifest "${merged.modelId}" has output.embeddingPostprocessor.projections[${i}].inputSize=${projection.inputSize}, ` +
          `expected ${expectedInputSize}.`
        );
      }
      expectedInputSize = projection.outputSize;
    }
  }

  // Get MoE config
  const moeConfig = manifest.moeConfig ?? null;
  const useMoE = (moeConfig?.numExperts ?? 0) > 1;
  if (useMoE && (moeConfig?.numExperts == null || moeConfig?.numExpertsPerToken == null || !moeConfig?.expertFormat)) {
    throw new Error(`Manifest "${manifest.modelId}" is missing moeConfig fields for MoE inference.`);
  }
  const numExperts = useMoE ? moeConfig.numExperts : 0;
  const moeTopK = useMoE ? moeConfig.numExpertsPerToken : 0;
  const expertFormat = useMoE ? moeConfig.expertFormat : null;
  const moeExpertIntermediateSize = useMoE && moeConfig.expertIntermediateSize != null
    ? Number(moeConfig.expertIntermediateSize)
    : resolvedIntermediateSize;
  if (
    useMoE
    && expertFormat === 'gemma4'
    && (!Number.isFinite(moeExpertIntermediateSize) || moeExpertIntermediateSize <= 0)
  ) {
    throw new Error(`Manifest "${manifest.modelId}" has invalid moeConfig.expertIntermediateSize for Gemma-style experts.`);
  }

  // RoPE scaling - use manifest inference as source of truth (not raw config)
  const ropeScale = inf.rope.ropeScalingFactor;
  const ropeScalingType = inf.rope.ropeScalingType;
  const ropeLocalScale = inf.rope.ropeLocalScalingFactor;
  const ropeLocalScalingType = inf.rope.ropeLocalScalingType;
  const partialRotaryFactor = inf.rope.partialRotaryFactor;
  const ropeLocalPartialRotaryFactor = inf.rope.ropeLocalPartialRotaryFactor;
  const mropeInterleaved = inf.rope.mropeInterleaved === true;
  const ropeInterleaved = inf.rope.ropeInterleaved === true;

  if (ropeLocalScale == null && (inf.rope.ropeLocalTheta != null || inf.rope.mropeSection != null)) {
    throw new Error(
      `Model "${merged.modelId}" uses hybrid/mRoPE but is missing rope.ropeLocalScalingFactor in manifest. ` +
      `Re-convert the model using the latest converter or update the manifest to include an explicit scale.`
    );
  }
  const mropeSection = Array.isArray(inf.rope.mropeSection)
    ? inf.rope.mropeSection.map((entry) => Math.trunc(Number(entry)))
    : null;
  const ropeRotaryDim = resolveRotaryDim(archGlobalHeadDim ?? archHeadDim, partialRotaryFactor, merged.modelId);
  const ropeLocalRotaryDim = resolveRotaryDim(archHeadDim, ropeLocalPartialRotaryFactor, merged.modelId);
  const ropeFrequencyBaseDim = resolveFrequencyBaseDim(
    archGlobalHeadDim ?? archHeadDim,
    ropeRotaryDim,
    inf.rope.ropeFrequencyBaseDim,
    merged.modelId,
    'rope.ropeFrequencyBaseDim'
  );
  const ropeLocalFrequencyBaseDim = resolveFrequencyBaseDim(
    archHeadDim,
    ropeLocalRotaryDim,
    inf.rope.ropeLocalFrequencyBaseDim,
    merged.modelId,
    'rope.ropeLocalFrequencyBaseDim'
  );
  if (mropeSection && mropeSection.some((entry) => !Number.isFinite(entry) || entry <= 0)) {
    throw new Error(
      `Manifest "${merged.modelId}" has invalid rope.mropeSection; expected positive integers.`
    );
  }
  if (mropeInterleaved && mropeSection) {
    const doubledMropeDim = mropeSection.reduce((sum, entry) => sum + entry, 0) * 2;
    if (doubledMropeDim !== ropeRotaryDim) {
      throw new Error(
        `Manifest "${merged.modelId}" declares rope.mropeSection=${JSON.stringify(mropeSection)}, ` +
        `which expands to rotary dim ${doubledMropeDim}, but the resolved rotary dim is ${ropeRotaryDim}.`
      );
    }
  }
  const validateLongropeFactorArray = (value, label) => {
    if (!Array.isArray(value)) {
      throw new Error(`Manifest "${merged.modelId}" ${label} must be an array for LongRoPE.`);
    }
    if (value.length !== ropeRotaryDim / 2) {
      throw new Error(
        `Manifest "${merged.modelId}" ${label} length ${value.length} does not match ` +
        `resolved RoPE half dim ${ropeRotaryDim / 2}.`
      );
    }
    for (let i = 0; i < value.length; i += 1) {
      const entry = Number(value[i]);
      if (!Number.isFinite(entry) || entry <= 0) {
        throw new Error(`Manifest "${merged.modelId}" ${label}[${i}] must be a positive finite number.`);
      }
    }
    return value.map((entry) => Number(entry));
  };

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
    ...(ropeScalingType === 'longrope' && {
      short_factor: validateLongropeFactorArray(inf.rope.longropeShortFactor, 'rope.longropeShortFactor'),
      long_factor: validateLongropeFactorArray(inf.rope.longropeLongFactor, 'rope.longropeLongFactor'),
      original_max_position_embeddings: inf.rope.longropeOriginalMaxPos,
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
  validateChatTemplateType(chatTemplateType, merged.modelId);
  const chatTemplateEnabled = inf.chatTemplate.enabled;
  const chatTemplateThinking = inf.chatTemplate.thinking ?? null;
  const parsePositiveInt = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return null;
    return Math.trunc(num);
  };

  const resolveTensorOutputRows = (tensorInfo, hiddenSize) => {
    const shape = Array.isArray(tensorInfo?.shape) ? tensorInfo.shape : null;
    if (!shape || shape.length < 2) return null;
    const dim0 = Number(shape[0]);
    const dim1 = Number(shape[1]);
    if (!Number.isFinite(dim0) || !Number.isFinite(dim1)) return null;
    if (dim1 === hiddenSize) return Math.trunc(dim0);
    if (dim0 === hiddenSize) return Math.trunc(dim1);
    return null;
  };

  const deriveGlobalKVHeadsFromManifest = () => {
    if (!Array.isArray(layerTypes) || archGlobalHeadDim == null) {
      return null;
    }
    const tensors = manifest?.tensors && typeof manifest.tensors === 'object' ? manifest.tensors : null;
    if (!tensors) return null;
    const globalLayerIdx = layerTypes.findIndex((layerType) => {
      const normalized = typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
      return normalized && normalized !== 'sliding_attention' && normalized !== 'local_attention'
        && normalized !== 'local' && normalized !== 'sliding';
    });
    if (globalLayerIdx < 0) return null;
    const layerMarker = `.layers.${globalLayerIdx}.`;
    for (const [tensorName, tensorInfo] of Object.entries(tensors)) {
      if (!tensorName.includes(layerMarker) || !tensorName.includes('.self_attn.k_proj.weight')) {
        continue;
      }
      const rows = resolveTensorOutputRows(tensorInfo, arch.hiddenSize);
      if (rows != null && rows % archGlobalHeadDim === 0) {
        return rows / archGlobalHeadDim;
      }
    }
    return null;
  };

  const archNumGlobalKVHeads = parsePositiveInt(
    arch.numGlobalKeyValueHeads
      ?? arch.numGlobalKVHeads
      ?? config.num_global_key_value_heads
      ?? config.num_global_kv_heads
  ) ?? deriveGlobalKVHeadsFromManifest();

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
  const hasMixedAttentionGeometry = archGlobalHeadDim != null && archGlobalHeadDim !== archHeadDim;
  const hasSharedKvLayers = archNumKvSharedLayers > 0;
  const hasExplicitLayerTypes = Array.isArray(layerTypes) && layerTypes.length === arch.numLayers;
  const decodeStrategy = (hasMixedAttentionGeometry || hasSharedKvLayers) && !hasExplicitLayerTypes
    ? 'replay_prefill'
    : 'incremental';
  const perLayerInputsSession = resolvePerLayerInputsSession(inf, merged.modelId);
  const sessionSettings = resolveSessionSettings(inf, merged.modelId);
  const largeWeightsConfig = resolveLargeWeightsConfig(inf, merged.modelId);
  const diffusionGemma = inf.diffusionGemma ?? null;
  const embeddingScale = inf.output.embeddingScale;
  const logitInputScale = inf.output.logitInputScale;
  const residualBranchScale = inf.layerPattern.residualBranchScale;
  const heterogeneousAttention = resolveHeterogeneousAttentionContract(inf, arch.numLayers, merged.modelId);
  const postNormContract = resolvePostNormContract(inf.normalization);
  return {
    modelType: manifest.modelType,
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: resolvedIntermediateSize,
    intermediateSizes,
    maxIntermediateSize,
    numHeads: archNumHeads,
    numKVHeads: archNumKVHeads,
    numGlobalKVHeads: archNumGlobalKVHeads,
    headDim: archHeadDim,
    globalHeadDim: archGlobalHeadDim,
    vocabSize: arch.vocabSize,
    hiddenSizePerLayerInput: arch.hiddenSizePerLayerInput ?? null,
    vocabSizePerLayerInput: arch.vocabSizePerLayerInput ?? null,
    numKvSharedLayers: archNumKvSharedLayers,
    maxSeqLen: arch.maxSeqLen,
    useMoE,
    numExperts,
    moeTopK,
    expertFormat,
    moeExpertIntermediateSize,
    slidingWindow: inf.attention.slidingWindow,
    ropeTheta: inf.rope.ropeTheta,
    ropeLocalTheta: inf.rope.ropeLocalTheta,
    ropeRotaryDim,
    ropeLocalRotaryDim,
    ropeFrequencyBaseDim,
    ropeLocalFrequencyBaseDim,
    ropeInterleaved,
    mropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeLocalPartialRotaryFactor,
    ropeScale,
    ropeLocalScale,
    ropeScalingType,
    ropeLocalScalingType,
    ropeScaling,
    ropeLocalScaling,
    quantization: manifest.quantization,
    quantMethod: config.quantization_config?.quant_method ?? null,
    normalizationType: inf.normalization.type ?? 'rmsnorm',
    finalNormBiasTensor: inf.normalization.finalNormBiasTensor ?? null,
    rmsNormEps: inf.normalization.rmsNormEps,
    rmsNormWeightOffset: inf.normalization.rmsNormWeightOffset,
    ...postNormContract,
    postAttentionNorm: inf.normalization.postAttentionNorm,
    preFeedforwardNorm: inf.normalization.preFeedforwardNorm,
    postFeedforwardNorm: inf.normalization.postFeedforwardNorm,
    scaleEmbeddings: inf.output.scaleEmbeddings,
    embeddingScale,
    embeddingNormalization: resolveEmbeddingNormalization(inf.output.embeddingNormalization),
    logitInputScale,
    logitOutputScale: inf.output.logitOutputScale ?? 1,
    residualBranchScale,
    useTiedEmbeddings: inf.output.tieWordEmbeddings,
    embeddingTranspose: inf.output.embeddingTranspose,
    embeddingVocabSize: inf.output.embeddingVocabSize,
    embeddingPostprocessor,
    hiddenActivation,
    gatedActivation: inf.ffn.gatedActivation,
    ffnBranchMode: inf.ffn.branchMode,
    useDoubleWideMlp: inf.ffn.useDoubleWideMlp,
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
    queryKeyNormType: inf.attention.queryKeyNormType ?? 'rmsnorm',
    queryKeyNormAxis: inf.attention.queryKeyNormAxis ?? 'head',
    queryKeyNormLayers: inf.attention.queryKeyNormLayers ?? null,
    queryKeyNormWeightLayers: inf.attention.queryKeyNormWeightLayers ?? null,
    valueNorm: inf.attention.valueNorm,
    attentionOutputGate: inf.attention.attentionOutputGate === true,
    outputGateType: inf.attention.outputGateType ?? null,
    queryPreAttnScalar,
    ...heterogeneousAttention,
    layerPipeline: inf.pipeline ?? null,
    chatTemplateType,
    chatTemplateEnabled,
    chatTemplateThinking,
    decodeStrategy,
    diffusionGemma,
    perLayerInputsSession,
    sessionSettings,
    largeWeightsConfig,
    kernelPath: null,
    visionConfig: resolveVisionConfig(config, manifest),
    audioConfig: resolveAudioConfig(config, manifest),
  };
}

export function parseModelConfigFromManifest(manifest, runtimeOverrides) {
  assertSupportedRuntimeModelType(manifest);
  validateModelOverrides(runtimeOverrides, 'runtime.inference.modelOverrides');

  // Merge manifest inference with runtime overrides
  const merged = mergeConfig(
    {
      modelId: manifest.modelId ?? 'unknown',
      inference: manifest.inference,
      architecture: manifest.architecture,
      config: manifest.config ?? null,
      quantization_config: manifest.quantization_config ?? null,
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

  // Dump full field-to-source mapping at debug level for diagnostics
  const sourceDump = dumpConfigSources(merged);
  log.debug('Config', `Config source map: ${JSON.stringify(sourceDump)}`);

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
      `Legacy family-registry resolution has been removed.`
    );
  }

  assertSupportedManifestInference(manifest);

  log.info('Config', 'Using manifest-first config (source of truth)');
  return parseModelConfigFromManifest(manifest, runtimeOverrides);
}
