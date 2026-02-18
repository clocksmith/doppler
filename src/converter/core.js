

import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';

import { classifyTensorRole, generateShardFilename } from '../storage/rdrr-format.js';
import { log } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import {
  createConverterConfig,
  detectPreset,
  listPresets,
  resolvePreset,
} from '../config/index.js';
import { buildManifestInference, inferEmbeddingOutputConfig } from './manifest-inference.js';
import { resolveEosTokenId } from './tokenizer-utils.js';
import {
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
} from './quantizer.js';

const CONVERSION_SUPPORTED_PRESETS = [...listPresets()]
  .filter((presetId) => !['transformer', 'diffusion'].includes(presetId))
  .sort()
  .join(', ');

// ============================================================================
// Re-exports for Backward Compatibility
// ============================================================================


export const ConvertStage = SchemaConversionStage;

// Re-export constants
export const SHARD_SIZE = SCHEMA_SHARD_SIZE;
export const RDRR_VERSION = SCHEMA_RDRR_VERSION;

// ============================================================================
// Pure Functions (no I/O, no platform dependencies)
// ============================================================================

function resolveTokenizerId(value) {
  if (typeof value === 'number') return value;
  return null;
}

function resolveTokenizerIds(value) {
  if (Array.isArray(value) && value.every((id) => typeof id === 'number')) {
    return value;
  }
  if (typeof value === 'number') return [value];
  return null;
}

function resolveTokenizerField(tokenizerConfig, ...keys) {
  if (!tokenizerConfig) return null;
  for (const key of keys) {
    if (tokenizerConfig[key] != null) {
      return tokenizerConfig[key];
    }
  }
  return null;
}

function resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture) {
  const configVocab = rawConfig?.vocab_size ?? rawConfig?.text_config?.vocab_size;
  const tokenizerVocab = tokenizerConfig?.vocab_size ?? tokenizerConfig?.vocabSize;
  const archVocab = architecture?.vocabSize;
  return tokenizerVocab ?? configVocab ?? archVocab ?? null;
}

function normalizeStorageQuant(value) {
  if (value == null) return null;
  const lower = String(value).trim().toLowerCase();
  if (!lower) return null;
  if (lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'fp32' || lower === 'float32') return 'f32';
  if (lower === 'bfloat16') return 'bf16';
  if (lower === 'q4_k_m' || lower === 'q4km') return 'q4k';
  return lower;
}

function resolveTensorTargetQuant(tensorName, fallbackQuant, quantizationInfo) {
  const fallback = normalizeStorageQuant(fallbackQuant);
  if (!quantizationInfo || typeof quantizationInfo !== 'object') {
    return fallback;
  }

  const role = classifyTensorRole(tensorName);
  if (role === 'embedding') {
    return normalizeStorageQuant(quantizationInfo.embeddings ?? fallback) ?? fallback;
  }
  if (role === 'lm_head') {
    const headQuant = quantizationInfo.lmHead ?? quantizationInfo.embeddings ?? fallback;
    return normalizeStorageQuant(headQuant) ?? fallback;
  }
  return normalizeStorageQuant(quantizationInfo.weights ?? fallback) ?? fallback;
}

function bf16ToFloat32(value) {
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, (value & 0xffff) << 16, true);
  return view.getFloat32(0, true);
}

function normalizeQ4KLayout(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized === 'col' ? 'col' : 'row';
}

function toFloat32ForQ4K(tensorData, sourceDtype, tensorName) {
  const dtype = String(sourceDtype || '').toUpperCase();
  if (dtype === 'F32') {
    if (tensorData.byteLength % 4 !== 0) {
      throw new Error(`Invalid F32 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    return new Float32Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 4
    );
  }
  if (dtype === 'F16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid F16 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    const f16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i++) {
      f32[i] = float16ToFloat32(f16[i]);
    }
    return f32;
  }
  if (dtype === 'BF16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid BF16 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    const bf16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(bf16.length);
    for (let i = 0; i < bf16.length; i++) {
      f32[i] = bf16ToFloat32(bf16[i]);
    }
    return f32;
  }
  throw new Error(`Cannot quantize ${tensorName} from ${dtype} to Q4_K_M`);
}

function resolveConfigTokenId(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = rawConfig?.text_config?.[key];
  return resolveTokenizerId(direct ?? nested);
}

function resolveConfigTokenIds(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = rawConfig?.text_config?.[key];
  return resolveTokenizerIds(direct ?? nested);
}

function resolveMoEConfigNumber(rawConfig, ...keys) {
  for (const key of keys) {
    const direct = rawConfig?.[key];
    if (Number.isFinite(direct) && direct > 0) return Number(direct);
    const nested = rawConfig?.text_config?.[key];
    if (Number.isFinite(nested) && nested > 0) return Number(nested);
  }
  return null;
}

function modelHasMoETensors(model) {
  if (!Array.isArray(model?.tensors)) return false;
  return model.tensors.some((tensor) => {
    const name = String(tensor?.name || '').toLowerCase();
    return (
      name.includes('.experts.') ||
      name.includes('.expert.') ||
      name.includes('block_sparse_moe')
    );
  });
}

function resolveMoEExpertFormat(rawConfig, resolvedModelType, quantizationInfo, explicitFormat) {
  if (explicitFormat) return explicitFormat;
  const fromQuant = quantizationInfo?.expertsFormat;
  if (typeof fromQuant === 'string' && fromQuant.length > 0) {
    return fromQuant;
  }
  const modelType = String(
    resolvedModelType ??
    rawConfig?.model_type ??
    rawConfig?.text_config?.model_type ??
    ''
  ).toLowerCase();
  if (modelType.includes('gpt_oss') || modelType.includes('gpt-oss') || modelType.includes('gptoss')) {
    return 'gpt-oss';
  }
  return 'mixtral';
}

function normalizeMoEConfig(config, contextLabel) {
  if (!config) return null;
  const numExperts = Number(config.numExperts);
  const numExpertsPerToken = Number(config.numExpertsPerToken);
  const expertFormat = String(config.expertFormat || '').trim();
  const allowedExpertFormat = expertFormat === 'gpt-oss' || expertFormat === 'mixtral';
  if (!Number.isFinite(numExperts) || numExperts <= 0) {
    throw new Error(`Invalid moeConfig.numExperts for ${contextLabel}`);
  }
  if (!Number.isFinite(numExpertsPerToken) || numExpertsPerToken <= 0) {
    throw new Error(`Invalid moeConfig.numExpertsPerToken for ${contextLabel}`);
  }
  if (numExpertsPerToken > numExperts) {
    throw new Error(`Invalid moeConfig for ${contextLabel}: numExpertsPerToken cannot exceed numExperts`);
  }
  if (!allowedExpertFormat) {
    throw new Error(`Invalid moeConfig.expertFormat for ${contextLabel}: "${expertFormat}"`);
  }
  return {
    numExperts,
    numExpertsPerToken,
    expertFormat,
  };
}

function resolveManifestMoEConfig(model, options, rawConfig, resolvedModelType) {
  const explicit = normalizeMoEConfig(options?.moeConfig ?? null, options?.modelId ?? 'model');
  if (explicit) return explicit;

  const hasMoETensors = modelHasMoETensors(model);
  const numExperts = resolveMoEConfigNumber(rawConfig, 'num_local_experts', 'num_experts', 'expertCount');

  // If the checkpoint does not expose MoE tensors and config does not declare experts,
  // this is a dense model and should not emit moeConfig.
  if (!hasMoETensors && (!numExperts || numExperts <= 1)) {
    return null;
  }

  if (!numExperts || numExperts <= 0) {
    throw new Error(
      `MoE tensors detected for "${options?.modelId ?? 'model'}" but expert count is missing in config`
    );
  }

  const numExpertsPerToken = resolveMoEConfigNumber(
    rawConfig,
    'num_experts_per_tok',
    'num_experts_per_token',
    'experts_per_token',
    'expertUsedCount'
  );

  if (!numExpertsPerToken) {
    throw new Error(
      `MoE model "${options?.modelId ?? 'model'}" missing experts-per-token config ` +
      '(expected num_experts_per_tok/num_experts_per_token/experts_per_token)'
    );
  }

  const expertFormat = resolveMoEExpertFormat(
    rawConfig,
    resolvedModelType,
    options?.quantizationInfo ?? null,
    null
  );

  return normalizeMoEConfig(
    { numExperts, numExpertsPerToken, expertFormat },
    options?.modelId ?? 'model'
  );
}

function buildSentencepieceTokenizer(tokenizerConfig, rawConfig, architecture, modelTokenizerModel) {
  if (!modelTokenizerModel) return null;

  const vocabSize = resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture);
  const sentencepieceModel = typeof modelTokenizerModel === 'string'
    ? modelTokenizerModel
    : modelTokenizerModel?.file ?? 'tokenizer.model';

  const bosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'bos_token_id', 'bosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'bos_token_id')
  );
  const eosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'eos_token_id', 'eosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'eos_token_id')
  );
  const eosTokens = resolveTokenizerIds(
    resolveTokenizerField(tokenizerConfig, 'eos_token_ids', 'eosTokens', 'eos_token_id')
    ?? resolveConfigTokenIds(rawConfig, 'eos_token_ids')
  );
  const padTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'pad_token_id', 'padTokenId')
    ?? resolveConfigTokenId(rawConfig, 'pad_token_id')
  );
  const unkTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'unk_token_id', 'unkTokenId')
    ?? resolveConfigTokenId(rawConfig, 'unk_token_id')
  );
  const addBosToken = resolveTokenizerField(tokenizerConfig, 'add_bos_token', 'addBosToken');
  const addEosToken = resolveTokenizerField(tokenizerConfig, 'add_eos_token', 'addEosToken');

  const tokenizer = {
    type: 'sentencepiece',
    sentencepieceModel,
    vocabSize: vocabSize ?? 0,
  };

  if (bosTokenId != null) tokenizer.bosTokenId = bosTokenId;
  if (eosTokenId != null) tokenizer.eosTokenId = eosTokenId;
  if (eosTokens) tokenizer.eosTokens = eosTokens;
  if (padTokenId != null) tokenizer.padTokenId = padTokenId;
  if (unkTokenId != null) tokenizer.unkTokenId = unkTokenId;
  if (addBosToken != null) tokenizer.addBosToken = addBosToken;
  if (addEosToken != null) tokenizer.addEosToken = addEosToken;

  return tokenizer;
}


export function sanitizeModelId(name) {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 64);
  return sanitized || null;
}


// Re-export formatBytes from schema for backward compatibility
export { formatBytes };

const BF16_ROUND_VIEW = new DataView(new ArrayBuffer(4));

function float32ToBFloat16(value) {
  BF16_ROUND_VIEW.setFloat32(0, value, true);
  const bits = BF16_ROUND_VIEW.getUint32(0, true);
  const lsb = (bits >> 16) & 1;
  const roundingBias = 0x7fff + lsb;
  return ((bits + roundingBias) >> 16) & 0xffff;
}


export function shouldQuantize(tensorName, shape, options = {}) {
  if (!shape || !Array.isArray(shape) || shape.length === 0) {
    log.warn('Convert', `Invalid shape for tensor "${tensorName}": ${JSON.stringify(shape)}`);
    return false;
  }
  const numElements = shape.reduce((a, b) => a * b, 1);
  const role = classifyTensorRole(tensorName);
  const lower = tensorName.toLowerCase();
  const isBias = lower.endsWith('.bias') || lower.endsWith('_bias');
  const quantizeEmbeddings = options.quantizeEmbeddings ?? false;

  return selectRuleValue('converter', 'tensorRoles', 'shouldQuantize', {
    numElements,
    role,
    isBias,
    quantizeEmbeddings,
  });
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

  // Try HuggingFace config first
  if (config && Object.keys(config).length > 0) {
    const numLayers = requireNumber(
      firstNumber(config.num_hidden_layers, config.n_layer, config.num_layers),
      'num_hidden_layers'
    );
    const hiddenSize = requireNumber(
      firstNumber(config.hidden_size, config.n_embd, config.embedding_size),
      'hidden_size'
    );
    const intermediateSize = requireNumber(
      firstNumber(config.intermediate_size, config.n_inner, config.ffn_dim),
      'intermediate_size'
    );
    const numHeads = requireNumber(
      firstNumber(config.num_attention_heads, config.n_head, config.attention_heads),
      'num_attention_heads'
    );
    const numKVHeads = firstNumber(config.num_key_value_heads, config.num_kv_heads) ?? numHeads;
    const headDimFromConfig = config.head_dim ?? Math.floor(hiddenSize / numHeads);
    const vocabSize = requireNumber(
      firstNumber(config.vocab_size, config.n_vocab),
      'vocab_size'
    );
    const maxSeqLen = requireNumber(
      firstNumber(config.max_position_embeddings, config.n_positions, config.max_seq_len),
      'max_position_embeddings'
    );
    const ropeTheta = config.rope_theta ?? undefined;

    return {
      numLayers,
      hiddenSize,
      intermediateSize,
      numAttentionHeads: numHeads,
      numKeyValueHeads: numKVHeads,
      headDim: headDimFromConfig,
      vocabSize,
      maxSeqLen,
      ropeTheta,
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

    return {
      numLayers,
      hiddenSize,
      intermediateSize,
      numAttentionHeads: numHeads,
      numKeyValueHeads: numKVHeads,
      headDim: Math.floor(hiddenSize / numHeads),
      vocabSize,
      maxSeqLen,
    };
  }

  throw new Error('Missing model config: cannot extract architecture');
}


export function buildTensorMap(tensors, shardSize) {
  if (!shardSize || shardSize <= 0) {
    throw new Error('Missing shard size for tensor map');
  }
  const tensorMap = {};

  let globalOffset = 0;
  for (const tensor of tensors) {
    const startShard = Math.floor(globalOffset / shardSize);
    const offsetInShard = globalOffset % shardSize;

    if (offsetInShard + tensor.size <= shardSize) {
      // Fits in single shard
      tensorMap[tensor.name] = {
        shard: startShard,
        offset: offsetInShard,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    } else {
      // Spans multiple shards
      const spans = [];
      let remaining = tensor.size;
      let currentShard = startShard;
      let currentOffset = offsetInShard;

      while (remaining > 0) {
        const available = shardSize - currentOffset;
        const chunkSize = Math.min(remaining, available);
        spans.push({
          shardIndex: currentShard,
          offset: currentOffset,
          size: chunkSize,
        });
        remaining -= chunkSize;
        currentShard++;
        currentOffset = 0;
      }

      tensorMap[tensor.name] = {
        spans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    }

    globalOffset += tensor.size;
  }

  return tensorMap;
}


export function createManifest(
  modelId,
  model,
  shards,
  tensorLocations,
  sourceOrOptions
) {
  if (!sourceOrOptions) {
    throw new Error('Missing manifest options');
  }
  const options = typeof sourceOrOptions === 'string' ? { source: sourceOrOptions } : sourceOrOptions ?? {};
  const source = options.source;
  if (!source) {
    throw new Error('Missing manifest source');
  }
  const resolvedModelType =
    options.modelType ??
    model.modelType ??
    model.config?.architectures?.[0] ??
    model.architecture;
  if (!resolvedModelType) {
    throw new Error('Missing modelType for manifest');
  }
  const isDiffusion = resolvedModelType === 'diffusion';
  const architecture = options.architecture ?? model.architecture ?? (
    isDiffusion ? 'diffusion' : extractArchitecture(model.config, model.ggufConfig)
  );
  const rawConfig = model.config || {};
  const moeConfig = isDiffusion
    ? null
    : resolveManifestMoEConfig(model, { ...options, modelId }, rawConfig, resolvedModelType);
  let inference = options.inference;
  if (!inference) {
    if (isDiffusion) {
      inference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    } else {
      const presetId = detectPreset(rawConfig, model.architecture);
      if (presetId === 'transformer') {
        const modelType = rawConfig.model_type ?? 'unknown';
        throw new Error(
          `Unknown model family: architecture="${model.architecture || 'unknown'}", model_type="${modelType}"\n\n` +
          `DOPPLER requires a known model preset to generate correct inference config.\n` +
          `The manifest-first architecture does not support generic defaults.\n\n` +
          `Options:\n` +
          `  1. Wait for official support of this model family\n` +
          `  2. Create a custom preset in src/config/presets/models/\n` +
          `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n` +
          `Supported model families: ${CONVERSION_SUPPORTED_PRESETS}`
        );
      }
      const preset = resolvePreset(presetId);
      const headDim = rawConfig.head_dim ?? (architecture && typeof architecture === 'object' ? architecture.headDim : null);
      if (!headDim) {
        throw new Error('Missing headDim in architecture');
      }
      const tensorNames = Array.isArray(model.tensors)
        ? model.tensors.map((tensor) => tensor.name)
        : null;
      inference = buildManifestInference(preset, rawConfig, headDim, options.quantizationInfo ?? null, tensorNames);
    }
  }

  const embeddingOutput = inferEmbeddingOutputConfig(tensorLocations);
  if (embeddingOutput) {
    inference = {
      ...inference,
      output: {
        ...inference.output,
        ...embeddingOutput,
      },
    };
  }

  const eosTokenId = options.eosTokenId !== undefined
    ? options.eosTokenId
    : isDiffusion
      ? null
      : resolveEosTokenId({
          config: rawConfig,
          tokenizer: model.tokenizer ?? model.tokenizerConfig ?? null,
          tokenizerJson: model.tokenizerJson ?? null,
        });
  const resolvedQuantization = options.quantization ?? model.quantization;
  if (!resolvedQuantization) {
    throw new Error('Missing quantization for manifest');
  }
  const hashAlgorithm = options.hashAlgorithm;
  if (!hashAlgorithm) {
    throw new Error('Missing hashAlgorithm for manifest');
  }

  const manifest = {
    version: RDRR_VERSION,
    modelId,
    modelType: resolvedModelType,
    quantization: resolvedQuantization,
    quantizationInfo: options.quantizationInfo ?? undefined,
    architecture,
    moeConfig,
    inference,
    shards,
    tensors: tensorLocations,
    totalSize: shards.reduce((sum, s) => sum + s.size, 0),
    hashAlgorithm,
    eos_token_id: eosTokenId,
    config: isDiffusion ? rawConfig : undefined,
    metadata: {
      source,
      convertedAt: new Date().toISOString(),
    },
  };

  // Include tokenizer if available
  if (model.tokenizerJson) {
    const tokenizer = model.tokenizerJson;
    const vocabSize =
      tokenizer.model?.vocab?.length ||
      Object.keys(tokenizer.model?.vocab || {}).length;
    if (!vocabSize) {
      throw new Error('Tokenizer vocab is missing or empty');
    }
    manifest.tokenizer = {
      type: 'bundled',
      vocabSize,
    };
    manifest.metadata.hasTokenizer = true;
  } else {
    const tokenizer = buildSentencepieceTokenizer(
      model.tokenizerConfig ?? null,
      rawConfig,
      architecture,
      model.tokenizerModel ?? null
    );
    if (tokenizer) {
      manifest.tokenizer = tokenizer;
      manifest.metadata.hasTokenizer = true;
    }
  }

  return manifest;
}

// ============================================================================
// Main Converter (uses I/O adapter)
// ============================================================================


export async function convertModel(model, io, options = {}) {
  const { onProgress, signal } = options;
  const converterConfig = options.converterConfig || createConverterConfig();
  const shardSize = options.shardSize ?? converterConfig.sharding.shardSizeBytes;
  if (!shardSize || shardSize <= 0) {
    throw new Error('Missing shardSize for conversion');
  }
  const modelIdInput = options.modelId ?? converterConfig.output.modelId ?? model.modelId ?? model.name;
  const modelId = modelIdInput ? sanitizeModelId(modelIdInput) : null;
  if (!modelId) {
    throw new Error('Missing modelId for conversion');
  }
  const tensors = model.tensors;
  const totalTensors = tensors.length;
  const targetQuant = String(options.quantization ?? model.quantization ?? '').trim().toLowerCase();
  const q4kLayout = normalizeQ4KLayout(options.quantizationInfo?.layout);
  const quantizeEmbeddings = (
    normalizeStorageQuant(options.quantizationInfo?.embeddings ?? null) === 'q4k'
    || normalizeStorageQuant(options.quantizationInfo?.lmHead ?? null) === 'q4k'
  );
  const shards = [];
  const tensorLocations = {};

  // Current shard accumulator
  let currentShardIndex = 0;
  let currentShardData = [];
  let currentShardSize = 0;
  let totalSize = 0;
  let globalOffset = 0;

  // Helper to flush current shard
  const flushShard = async () => {
    if (currentShardData.length === 0) return;

    // Concatenate chunks
    const shardTotalSize = currentShardData.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(shardTotalSize);
    let offset = 0;
    for (const chunk of currentShardData) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    // Write shard and get hash
    const hash = await io.writeShard(currentShardIndex, shardData);

    shards.push({
      index: currentShardIndex,
      filename: generateShardFilename(currentShardIndex),
      size: shardData.length,
      hash,
      offset: currentShardIndex * shardSize,
    });

    currentShardIndex++;
    currentShardData = [];
    currentShardSize = 0;
  };

  // Process tensors
  for (let i = 0; i < tensors.length; i++) {
    if (signal?.aborted) {
      throw new DOMException('Conversion cancelled', 'AbortError');
    }

    const tensor = tensors[i];

    onProgress?.({
      stage: ConvertStage.WRITING,
      message: `Processing ${tensor.name}`,
      current: i + 1,
      total: totalTensors,
      percent: Math.round(((i + 1) / totalTensors) * 100),
    });

    // Read tensor data
    const data = await io.readTensorData(tensor);
    let tensorData = new Uint8Array(data);
    let outDtype = tensor.dtype;
    let outLayout = null;

    // Convert storage to requested format when needed so shard bytes stay consistent
    // with mixed-precision quantization metadata.
    const sourceDtype = String(tensor.dtype).toUpperCase();
    const tensorTargetQuant = resolveTensorTargetQuant(
      tensor.name,
      targetQuant,
      options.quantizationInfo ?? null
    );
    if (tensorTargetQuant === 'q4k') {
      const sourceQuant = normalizeStorageQuant(sourceDtype);
      if (sourceQuant === 'q4k') {
        outDtype = 'Q4_K_M';
        if (Array.isArray(tensor.shape) && tensor.shape.length === 2) {
          outLayout = q4kLayout;
        }
      } else if (shouldQuantize(tensor.name, tensor.shape, { quantizeEmbeddings })) {
        const f32Data = toFloat32ForQ4K(tensorData, sourceDtype, tensor.name);
        const quantized = (
          Array.isArray(tensor.shape) && tensor.shape.length === 2
            ? (q4kLayout === 'col'
              ? quantizeToQ4KMColumnWise(f32Data, tensor.shape)
              : quantizeToQ4KMRowWise(f32Data, tensor.shape))
            : quantizeToQ4KM(f32Data, tensor.shape)
        );
        tensorData = quantized.quantized;
        outDtype = 'Q4_K_M';
        if (Array.isArray(tensor.shape) && tensor.shape.length === 2) {
          outLayout = q4kLayout;
        }
      }
    } else if (tensorTargetQuant === 'f16' && sourceDtype === 'F32') {
      if (tensorData.byteLength % 4 !== 0) {
        throw new Error(`Invalid F32 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
      }
      const f32 = new Float32Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 4
      );
      const f16 = new Uint16Array(f32.length);
      for (let j = 0; j < f32.length; j++) {
        f16[j] = float32ToFloat16(f32[j]);
      }
      tensorData = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
      outDtype = 'F16';
    } else if (tensorTargetQuant === 'f16' && sourceDtype === 'BF16') {
      if (tensorData.byteLength % 2 !== 0) {
        throw new Error(`Invalid BF16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
      }
      const bf16 = new Uint16Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 2
      );
      const f16 = new Uint16Array(bf16.length);
      for (let j = 0; j < bf16.length; j++) {
        f16[j] = float32ToFloat16(bf16ToFloat32(bf16[j]));
      }
      tensorData = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
      outDtype = 'F16';
    } else if (tensorTargetQuant === 'bf16' && sourceDtype === 'F32') {
      if (tensorData.byteLength % 4 !== 0) {
        throw new Error(`Invalid F32 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
      }
      const f32 = new Float32Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 4
      );
      const bf16 = new Uint16Array(f32.length);
      for (let j = 0; j < f32.length; j++) {
        bf16[j] = float32ToBFloat16(f32[j]);
      }
      tensorData = new Uint8Array(bf16.buffer, bf16.byteOffset, bf16.byteLength);
      outDtype = 'BF16';
    } else if (tensorTargetQuant === 'f32' && sourceDtype === 'F16') {
      if (tensorData.byteLength % 2 !== 0) {
        throw new Error(`Invalid F16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
      }
      const f16 = new Uint16Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 2
      );
      const f32 = new Float32Array(f16.length);
      for (let j = 0; j < f16.length; j++) {
        f32[j] = float16ToFloat32(f16[j]);
      }
      tensorData = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
      outDtype = 'F32';
    } else if (tensorTargetQuant === 'f32' && sourceDtype === 'BF16') {
      if (tensorData.byteLength % 2 !== 0) {
        throw new Error(`Invalid BF16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
      }
      const bf16 = new Uint16Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 2
      );
      const f32 = new Float32Array(bf16.length);
      for (let j = 0; j < bf16.length; j++) {
        f32[j] = bf16ToFloat32(bf16[j]);
      }
      tensorData = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
      outDtype = 'F32';
    }

    const tensorSize = tensorData.byteLength;

    // Track tensor location
    const startShard = currentShardIndex;
    const offsetInShard = currentShardSize;
    const tensorSpans = [];

    // Add to current shard, splitting if necessary
    let remaining = tensorData;
    while (remaining.length > 0) {
      const availableInShard = shardSize - currentShardSize;
      const chunkSize = Math.min(remaining.length, availableInShard);

      currentShardData.push(remaining.slice(0, chunkSize));
      currentShardSize += chunkSize;
      totalSize += chunkSize;

      tensorSpans.push({
        shardIndex: currentShardIndex,
        offset: currentShardSize - chunkSize,
        size: chunkSize,
      });

      remaining = remaining.slice(chunkSize);

      // Flush shard if full
      if (currentShardSize >= shardSize) {
        await flushShard();
      }
    }

    // Record tensor location
    const role = classifyTensorRole(tensor.name);

    if (tensorSpans.length === 1) {
      tensorLocations[tensor.name] = {
        shard: tensorSpans[0].shardIndex,
        offset: tensorSpans[0].offset,
        size: tensorSize,
        shape: tensor.shape,
        dtype: outDtype,
        role,
        ...(outLayout ? { layout: outLayout } : {}),
      };
    } else {
      tensorLocations[tensor.name] = {
        spans: tensorSpans,
        size: tensorSize,
        shape: tensor.shape,
        dtype: outDtype,
        role,
        ...(outLayout ? { layout: outLayout } : {}),
      };
    }

    globalOffset += tensorSize;
  }

  // Flush final shard
  await flushShard();

  if (signal?.aborted) {
    throw new DOMException('Conversion cancelled', 'AbortError');
  }

  // Create manifest
  onProgress?.({
    stage: ConvertStage.MANIFEST,
    message: 'Creating manifest...',
  });

  const tensorEntries = Object.entries(tensorLocations).map(([name, location]) => ({
    name,
    dtype: location?.dtype ?? null,
    role: location?.role ?? null,
    layout: location?.layout ?? null,
  }));
  const effectiveQuantizationInfo = resolveEffectiveQuantizationInfo(
    options.quantizationInfo ?? null,
    tensorEntries
  );
  const effectiveManifestQuantization = resolveManifestQuantization(
    effectiveQuantizationInfo.weights,
    options.quantization ?? model.quantization
  );

  const manifest = createManifest(modelId, model, shards, tensorLocations, {
    source: 'convert-core',
    modelType: options.modelType,
    quantization: effectiveManifestQuantization,
    quantizationInfo: effectiveQuantizationInfo,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    architecture: options.architecture,
    inference: options.inference,
    eosTokenId: options.eosTokenId,
  });

  // Write manifest
  await io.writeManifest(manifest);

  onProgress?.({
    stage: ConvertStage.COMPLETE,
    message: 'Conversion complete!',
    modelId,
    shardCount: shards.length,
    totalSize: formatBytes(totalSize),
  });

  return {
    manifest,
    shardCount: shards.length,
    tensorCount: tensors.length,
    totalSize,
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export { generateShardFilename };
