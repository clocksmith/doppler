

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
import { createConverterConfig, detectPreset, resolvePreset } from '../config/index.js';
import { buildManifestInference, inferEmbeddingOutputConfig } from './manifest-inference.js';
import { resolveEosTokenId } from './tokenizer-utils.js';

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
          `Supported model families: gemma2, gemma3, embeddinggemma, llama3, qwen3, mixtral, deepseek, mamba`
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
    const tensorData = new Uint8Array(data);

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
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        role,
      };
    } else {
      tensorLocations[tensor.name] = {
        spans: tensorSpans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        role,
      };
    }

    globalOffset += tensor.size;
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

  const manifest = createManifest(modelId, model, shards, tensorLocations, {
    source: 'convert-core',
    modelType: options.modelType,
    quantization: options.quantization,
    quantizationInfo: options.quantizationInfo,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
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
