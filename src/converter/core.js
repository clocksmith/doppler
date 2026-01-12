/**
 * core.ts - Platform-agnostic Model Conversion Core
 *
 * Shared types, pure functions, and conversion logic for RDRR format.
 * Used by both CLI (Node.js) and browser converters.
 *
 * Types are imported from config/schema for single source of truth.
 *
 * @module converter/core
 */

import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
} from '../config/schema/index.js';

import { generateShardFilename } from '../storage/rdrr-format.js';
import { log } from '../debug/index.js';
import { detectPreset, resolvePreset } from '../config/index.js';
import { buildManifestInference, inferEmbeddingOutputConfig } from './manifest-inference.js';

// ============================================================================
// Re-exports for Backward Compatibility
// ============================================================================

/** @deprecated Use ConversionStage from config/schema */
export const ConvertStage = SchemaConversionStage;

// Re-export constants
export const SHARD_SIZE = SCHEMA_SHARD_SIZE;
export const RDRR_VERSION = SCHEMA_RDRR_VERSION;

// ============================================================================
// Pure Functions (no I/O, no platform dependencies)
// ============================================================================

/**
 * Sanitize model ID for filesystem/URL safety
 */
export function sanitizeModelId(name) {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9_-]/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 64) || 'converted-model'
  );
}

/**
 * Format bytes for human-readable display
 */
export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/**
 * Check if tensor should be quantized based on name and shape
 */
export function shouldQuantize(tensorName, shape) {
  if (!shape || !Array.isArray(shape) || shape.length === 0) {
    log.warn('Convert', `Invalid shape for tensor "${tensorName}": ${JSON.stringify(shape)}`);
    return false;
  }
  const numElements = shape.reduce((a, b) => a * b, 1);
  if (numElements < 1024) return false;

  const lower = tensorName.toLowerCase();
  // Skip embeddings and output head (usually want full precision)
  if (lower.includes('embed') || lower.includes('lm_head')) return false;
  // Skip normalization layers (small, need precision)
  if (lower.includes('norm') || lower.includes('ln_')) return false;
  // Skip biases (small, need precision)
  if (lower.endsWith('.bias') || lower.endsWith('_bias')) return false;

  return true;
}

/**
 * Extract architecture configuration from model config
 */
export function extractArchitecture(config, ggufConfig) {
  // Try HuggingFace config first
  if (config && Object.keys(config).length > 0) {
    const numLayers = config.num_hidden_layers ?? config.n_layer ?? 32;
    const hiddenSize = config.hidden_size ?? config.n_embd ?? 4096;
    const intermediateSize = config.intermediate_size ?? config.n_inner ?? 11008;
    const numHeads = config.num_attention_heads ?? config.n_head ?? 32;
    const numKVHeads = config.num_key_value_heads ?? numHeads;
    const headDimFromConfig = config.head_dim ?? Math.floor(hiddenSize / numHeads);
    const vocabSize = config.vocab_size ?? 32000;
    const maxSeqLen = config.max_position_embeddings ?? config.n_positions ?? 2048;
    const ropeTheta = config.rope_theta ?? 10000;

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

  // Fallback for GGUF
  if (ggufConfig) {
    const c = ggufConfig;
    return {
      numLayers: c.blockCount ?? c.block_count ?? 32,
      hiddenSize: c.embeddingLength ?? c.embedding_length ?? 4096,
      intermediateSize: c.feedForwardLength ?? c.feed_forward_length ?? 11008,
      numAttentionHeads: c.attentionHeadCount ?? c.attention_head_count ?? 32,
      numKeyValueHeads: c.attentionHeadCountKV ?? c.attention_head_count_kv ?? 32,
      headDim: Math.floor(
        (c.embeddingLength ?? c.embedding_length ?? 4096) /
          (c.attentionHeadCount ?? c.attention_head_count ?? 32)
      ),
      vocabSize: c.vocabSize ?? c.vocab_size ?? 32000,
      maxSeqLen: c.contextLength ?? c.context_length ?? 2048,
    };
  }

  // Default fallback
  return {
    numLayers: 32,
    hiddenSize: 4096,
    intermediateSize: 11008,
    numAttentionHeads: 32,
    numKeyValueHeads: 32,
    headDim: 128,
    vocabSize: 32000,
    maxSeqLen: 2048,
  };
}

/**
 * Build tensor location map for manifest
 */
export function buildTensorMap(tensors, shardSize = SHARD_SIZE) {
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

/**
 * Create RDRR manifest from model info and shards
 */
export function createManifest(
  modelId,
  model,
  shards,
  tensorLocations,
  sourceOrOptions = 'convert-core'
) {
  const options = typeof sourceOrOptions === 'string' ? { source: sourceOrOptions } : sourceOrOptions ?? {};
  const source = options.source ?? 'convert-core';
  const architecture = extractArchitecture(model.config);
  const rawConfig = model.config || {};
  let inference = options.inference;
  if (!inference) {
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
        `Supported model families: gemma2, gemma3, llama3, qwen3, mixtral, deepseek, mamba`
      );
    }
    const preset = resolvePreset(presetId);
    const headDim = rawConfig.head_dim ??
      architecture.headDim ??
      Math.floor(architecture.hiddenSize / architecture.numAttentionHeads);
    inference = buildManifestInference(preset, rawConfig, headDim || 64, options.quantizationInfo ?? null);
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

  const manifest = {
    version: RDRR_VERSION,
    modelId,
    modelType: options.modelType || model.config?.architectures?.[0] || model.architecture || 'unknown',
    quantization: model.quantization || 'F16',
    quantizationInfo: options.quantizationInfo ?? undefined,
    architecture,
    inference,
    shards,
    tensors: tensorLocations,
    totalSize: shards.reduce((sum, s) => sum + s.size, 0),
    hashAlgorithm: 'sha256',
    metadata: {
      source,
      convertedAt: new Date().toISOString(),
    },
  };

  // Include tokenizer if available
  if (model.tokenizerJson) {
    const tokenizer = model.tokenizerJson;
    manifest.tokenizer = {
      type: 'bundled',
      vocabSize:
        tokenizer.model?.vocab?.length ||
        Object.keys(tokenizer.model?.vocab || {}).length ||
        architecture.vocabSize,
    };
    manifest.metadata.hasTokenizer = true;
  }

  return manifest;
}

// ============================================================================
// Main Converter (uses I/O adapter)
// ============================================================================

/**
 * Convert a parsed model to RDRR format
 */
export async function convertModel(model, io, options = {}) {
  const {
    modelId: userModelId,
    shardSize = SHARD_SIZE,
    onProgress,
    signal,
  } = options;

  const modelId = sanitizeModelId(userModelId || 'converted-model');
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
    if (tensorSpans.length === 1) {
      tensorLocations[tensor.name] = {
        shard: tensorSpans[0].shardIndex,
        offset: tensorSpans[0].offset,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    } else {
      tensorLocations[tensor.name] = {
        spans: tensorSpans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
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

  const manifest = createManifest(modelId, model, shards, tensorLocations);

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
