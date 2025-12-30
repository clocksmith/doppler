/**
 * convert-core.ts - Platform-agnostic Model Conversion Core
 *
 * Shared types, pure functions, and conversion logic for RDRR format.
 * Used by both CLI (Node.js) and browser converters.
 *
 * @module tools/convert-core
 */

import { SHARD_SIZE, RDRR_VERSION, generateShardFilename } from '../storage/rdrr-format.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Tensor information from parsed model
 */
export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  offset?: number;
  /** Platform-specific source reference (File, file handle, etc.) */
  _source?: unknown;
}

/**
 * Parsed model ready for conversion
 */
export interface ParsedModel {
  tensors: TensorInfo[];
  config: ModelConfig;
  architecture?: string;
  quantization?: string;
  tokenizerJson?: unknown;
}

/**
 * Model configuration from config.json or GGUF metadata
 */
export interface ModelConfig {
  architectures?: string[];
  model_type?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  rope_theta?: number;
  rms_norm_eps?: number;
  head_dim?: number;
  _name_or_path?: string;
  // GGUF-style fields
  n_layer?: number;
  n_embd?: number;
  n_head?: number;
  n_inner?: number;
  n_positions?: number;
  // Allow additional fields
  [key: string]: unknown;
}

/**
 * Conversion progress stages
 */
export const ConvertStage = {
  DETECTING: 'detecting',
  PARSING: 'parsing',
  QUANTIZING: 'quantizing',
  WRITING: 'writing',
  MANIFEST: 'manifest',
  COMPLETE: 'complete',
  ERROR: 'error',
} as const;

export type ConvertStageType = (typeof ConvertStage)[keyof typeof ConvertStage];

/**
 * Progress callback payload
 */
export interface ConvertProgress {
  stage: ConvertStageType;
  message: string;
  format?: string;
  modelId?: string;
  tensorCount?: number;
  totalSize?: string;
  current?: number;
  total?: number;
  percent?: number;
  shardCount?: number;
  error?: Error;
}

/**
 * Conversion options
 */
export interface ConvertOptions {
  modelId?: string;
  quantize?: 'q4_k_m' | 'f16' | 'f32' | null;
  quantizeEmbeddings?: boolean;
  shardSize?: number; // bytes, default from SHARD_SIZE
  onProgress?: (progress: ConvertProgress) => void;
  signal?: AbortSignal;
}

/**
 * Shard information for manifest
 */
export interface ShardInfo {
  index: number;
  filename: string;
  size: number;
  hash: string;
  offset: number;
}

/**
 * Tensor span for multi-shard tensors
 */
export interface TensorSpan {
  shardIndex: number;
  offset: number;
  size: number;
}

/**
 * Tensor location (single shard)
 */
export interface TensorLocationSingle {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
}

/**
 * Tensor location (multi shard)
 */
export interface TensorLocationMulti {
  spans: TensorSpan[];
  size: number;
  shape: number[];
  dtype: string;
}

export type TensorLocation = TensorLocationSingle | TensorLocationMulti;

/**
 * Architecture configuration
 */
export interface ArchitectureConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  ropeTheta?: number;
}

/**
 * Tokenizer info in manifest
 */
export interface TokenizerInfo {
  type: string;
  vocabSize: number;
}

/**
 * RDRR manifest structure
 */
export interface RDRRManifest {
  version: number | string;
  modelId: string;
  modelType: string;
  quantization: string;
  architecture: ArchitectureConfig;
  shards: ShardInfo[];
  tensors: Record<string, TensorLocation>;
  totalSize: number;
  hashAlgorithm: string;
  tokenizer?: TokenizerInfo;
  metadata: {
    source: string;
    convertedAt: string;
    hasTokenizer?: boolean;
  };
}

/**
 * Conversion result
 */
export interface ConvertResult {
  manifest: RDRRManifest;
  shardCount: number;
  tensorCount: number;
  totalSize: number;
}

/**
 * I/O Adapter interface - platform-specific implementations provide these
 */
export interface ConvertIO {
  /** Read tensor data from source */
  readTensorData(tensor: TensorInfo): Promise<ArrayBuffer>;
  /** Write shard data, returns hash */
  writeShard(index: number, data: Uint8Array): Promise<string>;
  /** Write manifest JSON */
  writeManifest(manifest: RDRRManifest): Promise<void>;
  /** Optional: compute hash (default: SHA-256 via crypto.subtle) */
  computeHash?(data: Uint8Array): Promise<string>;
}

// ============================================================================
// Pure Functions (no I/O, no platform dependencies)
// ============================================================================

/**
 * Sanitize model ID for filesystem/URL safety
 */
export function sanitizeModelId(name: string): string {
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
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/**
 * Check if tensor should be quantized based on name and shape
 */
export function shouldQuantize(tensorName: string, shape: number[]): boolean {
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
export function extractArchitecture(
  config: ModelConfig,
  ggufConfig?: Record<string, unknown>
): ArchitectureConfig {
  // Try HuggingFace config first
  if (config && Object.keys(config).length > 0) {
    const numLayers = (config.num_hidden_layers ?? config.n_layer ?? 32) as number;
    const hiddenSize = (config.hidden_size ?? config.n_embd ?? 4096) as number;
    const intermediateSize = (config.intermediate_size ?? config.n_inner ?? 11008) as number;
    const numHeads = (config.num_attention_heads ?? config.n_head ?? 32) as number;
    const numKVHeads = (config.num_key_value_heads ?? numHeads) as number;
    const headDimFromConfig = (config.head_dim ?? Math.floor(hiddenSize / numHeads)) as number;
    const vocabSize = (config.vocab_size ?? 32000) as number;
    const maxSeqLen = (config.max_position_embeddings ?? config.n_positions ?? 2048) as number;
    const ropeTheta = (config.rope_theta ?? 10000) as number;

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
      numLayers: (c.blockCount ?? c.block_count ?? 32) as number,
      hiddenSize: (c.embeddingLength ?? c.embedding_length ?? 4096) as number,
      intermediateSize: (c.feedForwardLength ?? c.feed_forward_length ?? 11008) as number,
      numAttentionHeads: (c.attentionHeadCount ?? c.attention_head_count ?? 32) as number,
      numKeyValueHeads: (c.attentionHeadCountKV ?? c.attention_head_count_kv ?? 32) as number,
      headDim: Math.floor(
        ((c.embeddingLength ?? c.embedding_length ?? 4096) as number) /
          ((c.attentionHeadCount ?? c.attention_head_count ?? 32) as number)
      ),
      vocabSize: (c.vocabSize ?? c.vocab_size ?? 32000) as number,
      maxSeqLen: (c.contextLength ?? c.context_length ?? 2048) as number,
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
export function buildTensorMap(
  tensors: Array<{ name: string; shape: number[]; dtype: string; size: number }>,
  shardSize: number = SHARD_SIZE
): Record<string, TensorLocation> {
  const tensorMap: Record<string, TensorLocation> = {};

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
      const spans: TensorSpan[] = [];
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
  modelId: string,
  model: ParsedModel,
  shards: ShardInfo[],
  tensorLocations: Record<string, TensorLocation>,
  source: string = 'convert-core'
): RDRRManifest {
  const architecture = extractArchitecture(model.config);

  const manifest: RDRRManifest = {
    version: RDRR_VERSION,
    modelId,
    modelType: model.config?.architectures?.[0] || model.architecture || 'unknown',
    quantization: model.quantization || 'F16',
    architecture,
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
    const tokenizer = model.tokenizerJson as {
      model?: { vocab?: unknown[] | Record<string, unknown> };
    };
    manifest.tokenizer = {
      type: 'bundled',
      vocabSize:
        (tokenizer.model?.vocab as unknown[])?.length ||
        Object.keys((tokenizer.model?.vocab as Record<string, unknown>) || {}).length ||
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
 *
 * @param model - Parsed model with tensors and config
 * @param io - Platform-specific I/O adapter
 * @param options - Conversion options
 * @returns Conversion result with manifest
 */
export async function convertModel(
  model: ParsedModel,
  io: ConvertIO,
  options: ConvertOptions = {}
): Promise<ConvertResult> {
  const {
    modelId: userModelId,
    shardSize = SHARD_SIZE,
    onProgress,
    signal,
  } = options;

  const modelId = sanitizeModelId(userModelId || 'converted-model');
  const tensors = model.tensors;
  const totalTensors = tensors.length;
  const shards: ShardInfo[] = [];
  const tensorLocations: Record<string, TensorLocation> = {};

  // Current shard accumulator
  let currentShardIndex = 0;
  let currentShardData: Uint8Array[] = [];
  let currentShardSize = 0;
  let totalSize = 0;
  let globalOffset = 0;

  // Helper to flush current shard
  const flushShard = async (): Promise<void> => {
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
    const tensorSpans: TensorSpan[] = [];

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

export { SHARD_SIZE, RDRR_VERSION, generateShardFilename };
