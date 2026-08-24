import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';
import {
  classifyTensorRole,
  generateShardFilename,
  resolveTensorGroup,
  resolveTensorRole,
} from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  getInferenceLayerPatternContractArtifact,
  selectRuleValue,
} from '../rules/rule-registry.js';
import {
  createConverterConfig,
} from '../config/index.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import { resolveEosTokenId } from './tokenizer-special-tokens.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';
import {
  normalizeQ4KLayout,
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  quantizeToInt4PerRowSymmetric,
} from './quantizer.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { SOURCE_PACKED_QUANT_DTYPES, isGemma4PerLayerEmbedTensor, normalizeModulesToNotConvert, normalizeStorageQuant, resolveQuantizeEmbeddings, transformTensorBytes } from './tensor-transform.js';

export function sanitizeModelId(name) {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 64);
  return sanitized || null;
}

export function extractArchitecture(config, ggufConfig) {
  const firstNumber = (...values) => {
    for (const value of values) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
      }
    }
    return null;
  };

  const requireNumber = (value, label) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`Missing ${label} in model config`);
    }
    return value;
  };

  const normalizeLinearNormMode = (value, sharedFlag = null) => {
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase();
      if (normalized === 'shared') return 'shared';
      if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
        return 'per_head';
      }
      throw new Error(
        `Unsupported linear_norm_mode="${value}" in model config. Supported values: "shared", "per_head".`
      );
    }
    if (typeof sharedFlag === 'boolean') {
      return sharedFlag ? 'shared' : 'per_head';
    }
    return undefined;
  };

  // Try HuggingFace config first
  if (config && Object.keys(config).length > 0) {
    const textConfig = getNestedTextConfig(config);
    const fromConfig = (...keys) => {
      const values = [];
      for (const key of keys) {
        values.push(config[key]);
      }
      for (const key of keys) {
        values.push(textConfig?.[key]);
      }
      return firstNumber(...values);
    };
    const fromConfigValue = (...keys) => {
      for (const key of keys) {
        if (config[key] !== undefined) return config[key];
      }
      for (const key of keys) {
        if (textConfig?.[key] !== undefined) return textConfig[key];
      }
      return undefined;
    };
    const numLayers = requireNumber(
      fromConfig('num_hidden_layers', 'n_layer', 'num_layers'),
      'num_hidden_layers'
    );
    const hiddenSize = requireNumber(
      fromConfig('hidden_size', 'n_embd', 'embedding_size'),
      'hidden_size'
    );
    const intermediateSize = requireNumber(
      fromConfig('intermediate_size', 'n_inner', 'ffn_dim'),
      'intermediate_size'
    );
    const numHeads = requireNumber(
      fromConfig('num_attention_heads', 'n_head', 'attention_heads'),
      'num_attention_heads'
    );
    const numKVHeads = fromConfig('num_key_value_heads', 'num_kv_heads') ?? numHeads;
    const numGlobalKVHeads = fromConfig('num_global_key_value_heads', 'num_global_kv_heads');
    const headDimFromConfig = fromConfig('head_dim') ?? Math.floor(hiddenSize / numHeads);
    const vocabSize = requireNumber(
      fromConfig('vocab_size', 'n_vocab'),
      'vocab_size'
    );
    const maxSeqLen = requireNumber(
      fromConfig('max_position_embeddings', 'n_positions', 'max_seq_len', 'max_length'),
      'max_position_embeddings'
    );
    const ropeTheta = fromConfig('rope_theta') ?? undefined;
    const linearNumKeyHeads = fromConfig('linear_num_key_heads');
    const linearNumValueHeads = fromConfig('linear_num_value_heads');
    const linearKeyHeadDim = fromConfig('linear_key_head_dim');
    const linearValueHeadDim = fromConfig('linear_value_head_dim');
    const linearConvKernelDim = fromConfig('linear_conv_kernel_dim');
    const hiddenSizePerLayerInput = fromConfig('hidden_size_per_layer_input');
    const vocabSizePerLayerInput = fromConfig('vocab_size_per_layer_input');
    const globalHeadDim = fromConfig('global_head_dim');
    const numKvSharedLayers = fromConfig('num_kv_shared_layers');
    const linearNormModeConfigured = normalizeLinearNormMode(
      fromConfigValue('linear_norm_mode'),
      fromConfigValue('linear_norm_shared')
    );
    const modelType = String(fromConfigValue('model_type') ?? '').trim().toLowerCase();
    const rawLayerTypes = fromConfigValue('layer_types');
    const layerTypes = Array.isArray(rawLayerTypes) ? rawLayerTypes : null;
    const hasLinearLayers = Array.isArray(layerTypes)
      && layerTypes.some((entry) => {
        const normalized = String(entry ?? '').trim().toLowerCase();
        return normalized === 'linear_attention'
          || normalized === 'linear'
          || normalized === 'gated_delta'
          || normalized === 'gated_delta_net';
      });
    const linearNormMode = linearNormModeConfigured
      ?? ((hasLinearLayers && modelType.startsWith('qwen')) ? 'shared' : undefined);

    return {
      numLayers,
      hiddenSize,
      intermediateSize,
      numAttentionHeads: numHeads,
      numKeyValueHeads: numKVHeads,
      numGlobalKeyValueHeads: numGlobalKVHeads ?? undefined,
      headDim: headDimFromConfig,
      vocabSize,
      maxSeqLen,
      ropeTheta,
      linearNumKeyHeads,
      linearNumValueHeads,
      linearKeyHeadDim,
      linearValueHeadDim,
      linearConvKernelDim,
      hiddenSizePerLayerInput,
      vocabSizePerLayerInput,
      globalHeadDim,
      numKvSharedLayers,
      linearNormMode,
    };
  }

  // GGUF config
  if (ggufConfig) {
    const c = ggufConfig;
    const numLayers = requireNumber(
      firstNumber(c.blockCount, c.block_count),
      'blockCount'
    );
    const hiddenSize = requireNumber(
      firstNumber(c.embeddingLength, c.embedding_length),
      'embeddingLength'
    );
    const intermediateSize = requireNumber(
      firstNumber(c.feedForwardLength, c.feed_forward_length),
      'feedForwardLength'
    );
    const numHeads = requireNumber(
      firstNumber(c.attentionHeadCount, c.attention_head_count),
      'attentionHeadCount'
    );
    const numKVHeads = firstNumber(c.attentionHeadCountKV, c.attention_head_count_kv) ?? numHeads;
    const vocabSize = requireNumber(
      firstNumber(c.vocabSize, c.vocab_size),
      'vocabSize'
    );
    const maxSeqLen = requireNumber(
      firstNumber(c.contextLength, c.context_length),
      'contextLength'
    );

    // Gemma 4-specific fields (optional — undefined on non-Gemma-4 GGUFs).
    // key_length is per-head; pick the larger of (key_length, key_length_swa)
    // for globalHeadDim and the smaller for headDim. Matches the mixed-geometry
    // KV cache expected by src/inference/pipelines/text/layer.js.
    const keyLen = firstNumber(c.attentionKeyLength);
    const keyLenSwa = firstNumber(c.attentionKeyLengthSwa);
    const headDimFromGguf = keyLenSwa != null && keyLen != null
      ? Math.min(keyLen, keyLenSwa)
      : (keyLenSwa ?? Math.floor(hiddenSize / numHeads));
    const globalHeadDim = (keyLen != null && keyLenSwa != null && keyLen !== keyLenSwa)
      ? Math.max(keyLen, keyLenSwa)
      : undefined;
    const numKvSharedLayers = firstNumber(c.numKvSharedLayers);
    const hiddenSizePerLayerInput = firstNumber(c.hiddenSizePerLayerInput);
    const vocabSizePerLayerInput = hiddenSizePerLayerInput != null ? vocabSize : undefined;

    return {
      numLayers,
      hiddenSize,
      intermediateSize,
      numAttentionHeads: numHeads,
      numKeyValueHeads: numKVHeads,
      headDim: headDimFromGguf,
      vocabSize,
      maxSeqLen,
      ...(globalHeadDim != null ? { globalHeadDim } : {}),
      ...(numKvSharedLayers != null ? { numKvSharedLayers } : {}),
      ...(hiddenSizePerLayerInput != null ? { hiddenSizePerLayerInput } : {}),
      ...(vocabSizePerLayerInput != null ? { vocabSizePerLayerInput } : {}),
    };
  }

  throw new Error('Missing model config: cannot extract architecture');
}

export function getNestedTextConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    return null;
  }
  if (config.text_config && typeof config.text_config === 'object' && !Array.isArray(config.text_config)) {
    return config.text_config;
  }
  if (config.language_config && typeof config.language_config === 'object' && !Array.isArray(config.language_config)) {
    return config.language_config;
  }
  return null;
}

export function isPlainRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

export function stripUndefined(value) {
  if (Array.isArray(value)) {
    return value.map(stripUndefined);
  }
  if (!isPlainRecord(value)) {
    return value;
  }
  const out = {};
  for (const key of Object.keys(value).sort()) {
    const item = value[key];
    if (item !== undefined) {
      out[key] = stripUndefined(item);
    }
  }
  return out;
}

export function canonicalJson(value) {
  return JSON.stringify(stripUndefined(value));
}

export function normalizeDigest(value, label) {
  const digest = typeof value === 'string' ? value.trim() : '';
  if (!digest) {
    throw new Error(`Missing ${label} digest`);
  }
  return digest.startsWith('sha256:') ? digest : `sha256:${digest}`;
}

export function digestSuffix(value, length = 12) {
  return String(value || '').replace(/^sha256:/, '').slice(0, length);
}

export async function hashArtifactValue(hashString, value, label) {
  if (typeof hashString !== 'function') {
    return null;
  }
  const digest = await hashString(canonicalJson(value));
  return normalizeDigest(digest, label);
}

export function resolveArtifactSourceFormat(options) {
  const explicit = typeof options?.sourceFormat === 'string' ? options.sourceFormat.trim() : '';
  if (explicit) return explicit;
  const sourcePath = typeof options?.sourcePath === 'string' ? options.sourcePath.trim().toLowerCase() : '';
  if (sourcePath.endsWith('.gguf')) return 'gguf';
  if (sourcePath.endsWith('.tflite')) return 'tflite';
  if (sourcePath.endsWith('.task')) return 'task';
  if (sourcePath.endsWith('.litertlm')) return 'litertlm';
  return 'safetensors';
}

export function resolveSourceCheckpointIdentity(explicit, options) {
  const sourceRepo = typeof explicit?.sourceRepo === 'string' && explicit.sourceRepo.trim()
    ? explicit.sourceRepo.trim()
    : null;
  const sourceRevision = typeof explicit?.sourceRevision === 'string' && explicit.sourceRevision.trim()
    ? explicit.sourceRevision.trim()
    : null;
  const sourcePath = typeof options?.sourcePath === 'string' && options.sourcePath.trim()
    ? options.sourcePath.trim()
    : null;
  const source = typeof options?.source === 'string' && options.source.trim()
    ? options.source.trim()
    : null;
  const sourceCheckpointId = typeof explicit?.sourceCheckpointId === 'string' && explicit.sourceCheckpointId.trim()
    ? explicit.sourceCheckpointId.trim()
    : (
        sourceRepo && sourceRevision
          ? `${sourceRepo}@${sourceRevision}`
          : (sourceRepo ?? source ?? sourcePath ?? null)
      );
  return {
    sourceCheckpointId,
    sourceRepo,
    sourceRevision,
  };
}

export function inferArtifactModalitySet(modelType, tensorLocations, converterConfig) {
  if (modelType === 'diffusion') {
    return ['image'];
  }
  const names = Object.keys(tensorLocations ?? {}).map((name) => name.toLowerCase());
  const modalities = new Set();
  if (modelType === 'embedding') {
    modalities.add('embedding');
  } else {
    modalities.add('text');
  }
  if (converterConfig?.output?.textOnly === true) {
    return [...modalities].sort();
  }
  if (names.some((name) => name.includes('vision') || name.includes('visual') || name.includes('image'))) {
    modalities.add('vision');
  }
  if (names.some((name) => name.includes('audio'))) {
    modalities.add('audio');
  }
  if (names.some((name) => name.includes('projector') || name.includes('mm_projector'))) {
    modalities.add('projector');
  }
  return [...modalities].sort();
}

export function resolveMaterializationProfile(quantizationInfo, inference) {
  const materialization = typeof inference?.session?.perLayerInputs?.materialization === 'string'
    ? inference.session.perLayerInputs.materialization
    : 'standard';
  const perLayerEmbeddings = typeof quantizationInfo?.perLayerEmbeddings === 'string'
    ? quantizationInfo.perLayerEmbeddings
    : null;
  return perLayerEmbeddings ? `${materialization}-${perLayerEmbeddings}` : materialization;
}

export async function buildArtifactIdentity(options) {
  const explicit = isPlainRecord(options?.explicitArtifactIdentity)
    ? options.explicitArtifactIdentity
    : {};
  const hashString = options?.hashString;
  if (typeof hashString !== 'function' && Object.keys(explicit).length > 0) {
    return stripUndefined(explicit);
  }
  if (typeof hashString !== 'function') {
    return null;
  }

  const sourceIdentity = resolveSourceCheckpointIdentity(explicit, options);
  const sourceFormat = typeof explicit.sourceFormat === 'string' && explicit.sourceFormat.trim()
    ? explicit.sourceFormat.trim()
    : resolveArtifactSourceFormat(options);
  const conversionConfigDigest = explicit.conversionConfigDigest
    ?? (options.conversionConfig
      ? await hashArtifactValue(hashString, options.conversionConfig, 'conversionConfig')
      : null);
  const shardSetHash = explicit.shardSetHash ?? explicit.weightPackHash
    ?? await hashArtifactValue(
      hashString,
      {
        hashAlgorithm: options.hashAlgorithm,
        shards: (options.shards ?? []).map((shard) => ({
          index: shard.index,
          filename: shard.filename,
          size: shard.size,
          hash: shard.hash,
          offset: shard.offset,
        })),
      },
      'shardSet'
    );
  const modalitySet = Array.isArray(explicit.modalitySet) && explicit.modalitySet.length > 0
    ? [...explicit.modalitySet]
    : inferArtifactModalitySet(options.modelType, options.tensorLocations, options.converterConfig);
  const materializationProfile = explicit.materializationProfile
    ?? resolveMaterializationProfile(options.quantizationInfo, options.inference);
  const weightPackInput = {
    sourceCheckpointId: sourceIdentity.sourceCheckpointId,
    sourceFormat,
    modelType: options.modelType,
    modalitySet,
    quantizationInfo: options.quantizationInfo,
    materializationProfile,
    shardSetHash,
    sharding: {
      shardSizeBytes: options.converterConfig?.sharding?.shardSizeBytes ?? null,
    },
    output: {
      textOnly: options.converterConfig?.output?.textOnly === true,
    },
  };
  const weightPackHash = explicit.weightPackHash
    ?? await hashArtifactValue(hashString, weightPackInput, 'weightPack');
  const weightPackId = explicit.weightPackId
    ?? `${sanitizeModelId(options.modelId) ?? 'model'}-wp-${digestSuffix(weightPackHash)}`;
  const manifestVariantHash = await hashArtifactValue(
    hashString,
    {
      weightPackId,
      modelType: options.modelType,
      inference: options.inference,
      config: options.manifestConfig ?? null,
    },
    'manifestVariant'
  );
  const manifestVariantId = explicit.manifestVariantId
    ?? `${sanitizeModelId(options.modelId) ?? 'model'}-mv-${digestSuffix(manifestVariantHash)}`;

  return stripUndefined({
    ...explicit,
    sourceCheckpointId: sourceIdentity.sourceCheckpointId,
    sourceRepo: sourceIdentity.sourceRepo ?? undefined,
    sourceRevision: sourceIdentity.sourceRevision ?? undefined,
    sourceFormat,
    conversionConfigPath: explicit.conversionConfigPath ?? options.conversionConfigPath ?? undefined,
    conversionConfigDigest: conversionConfigDigest ?? undefined,
    weightPackId,
    weightPackHash,
    shardSetHash,
    manifestVariantId,
    modalitySet,
    materializationProfile,
    artifactCompleteness: explicit.artifactCompleteness ?? 'complete',
  });
}
