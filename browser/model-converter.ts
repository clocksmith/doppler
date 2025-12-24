/**
 * model-converter.ts - Browser Model Converter
 *
 * Converts GGUF and safetensors models to RDRR format in the browser.
 * Supports:
 * - GGUF files (via gguf-importer.js)
 * - Single safetensors files
 * - Sharded HuggingFace models (multiple safetensors + config)
 * - Optional Q4_K_M quantization
 *
 * Output is written to OPFS (Origin Private File System).
 *
 * @module browser/model-converter
 */

import { parseGGUFHeader, GGUFParseResult } from './gguf-parser-browser.js';
import {
  parseSafetensorsFile,
  parseSafetensorsSharded,
  parseConfigJson,
  parseTokenizerJson,
  parseIndexJson,
  readTensorData,
  detectModelFormat,
  getAuxiliaryFiles,
  calculateTotalSize,
  SafetensorsParseResult,
  ModelFormat,
  AuxiliaryFiles,
  TensorInfo,
  ModelConfig,
} from './safetensors-parser-browser.js';
import {
  initOPFS,
  openModelDirectory,
  saveManifest,
  deleteModel,
} from '../storage/shard-manager.js';
import { SHARD_SIZE, RDRR_VERSION, generateShardFilename } from '../storage/rdrr-format.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Conversion stages
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
 * Conversion progress payload
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
  onProgress?: (progress: ConvertProgress) => void;
  signal?: AbortSignal;
}

/**
 * Shard info
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
 * Architecture config
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
 * RDRR manifest
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
 * GGUF model config extracted from header
 */
interface GGUFModelConfig {
  blockCount?: number;
  embeddingLength?: number;
  feedForwardLength?: number;
  attentionHeadCount?: number;
  attentionHeadCountKV?: number;
  vocabSize?: number;
  contextLength?: number;
  [key: string]: unknown;
}

/**
 * Internal tensor info - more flexible than SafetensorsTensor
 */
interface InternalTensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  dtypeOriginal?: string;
  dtypeId?: number;
  offset: number;
  size: number;
  elemSize?: number;
  file?: File;
  shardFile?: string;
}

/**
 * Extended model info for internal use
 */
interface ModelInfo {
  format?: string;
  tensors: InternalTensorInfo[];
  config?: ModelConfig | GGUFModelConfig;
  architecture?: string;
  quantization?: string;
  tensorDataOffset?: number;
  file?: File;
  tokenizerJson?: unknown;
}

/**
 * Write result
 */
interface WriteResult {
  totalSize: number;
  tensorLocations: Record<string, TensorLocation>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute SHA-256 hash
 */
async function computeSHA256(data: ArrayBuffer | Uint8Array): Promise<string> {
  const buffer = data instanceof ArrayBuffer ? data : data.buffer as ArrayBuffer;
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = new Uint8Array(hashBuffer);
  return Array.from(hashArray)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Float16 to Float32 conversion
 */
function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 31) {
    return frac ? NaN : sign ? -Infinity : Infinity;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * BFloat16 to Float32 conversion
 */
function bfloat16ToFloat32(bf16: number): number {
  const f32View = new Float32Array(1);
  const u32View = new Uint32Array(f32View.buffer);
  u32View[0] = bf16 << 16;
  return f32View[0];
}

/**
 * Convert typed array to Float32
 */
function convertToFloat32(data: ArrayBuffer, dtype: string): Float32Array {
  if (dtype === 'F32') {
    return new Float32Array(data);
  }

  if (dtype === 'F16') {
    const f16 = new Uint16Array(data);
    const f32 = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i++) {
      f32[i] = float16ToFloat32(f16[i]);
    }
    return f32;
  }

  if (dtype === 'BF16') {
    const bf16 = new Uint16Array(data);
    const f32 = new Float32Array(bf16.length);
    for (let i = 0; i < bf16.length; i++) {
      f32[i] = bfloat16ToFloat32(bf16[i]);
    }
    return f32;
  }

  throw new Error(`Unsupported dtype: ${dtype}`);
}

/**
 * Check if tensor should be quantized
 */
function shouldQuantize(tensorName: string, shape: number[]): boolean {
  const numElements = shape.reduce((a, b) => a * b, 1);
  if (numElements < 1024) return false;

  const lower = tensorName.toLowerCase();
  if (lower.includes('embed') || lower.includes('lm_head')) return false;
  if (lower.includes('norm') || lower.includes('ln_')) return false;
  if (lower.endsWith('.bias') || lower.endsWith('_bias')) return false;

  return true;
}

/**
 * Sanitize model ID
 */
function sanitizeModelId(name: string): string {
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
 * Format bytes for display
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// ============================================================================
// Main Convert Function
// ============================================================================

/**
 * Convert model files to RDRR format
 *
 * @param files - Selected model files
 * @param options - Conversion options
 * @returns Model ID
 */
export async function convertModel(files: File[], options: ConvertOptions = {}): Promise<string> {
  const { modelId: userModelId, quantize, onProgress, signal } = options;

  let modelId: string | null = null;
  let modelDir: FileSystemDirectoryHandle | null = null;
  const shardInfos: ShardInfo[] = [];

  try {
    // Initialize OPFS
    await initOPFS();

    // Detect format
    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: 'Detecting model format...',
    });

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    const format = detectModelFormat(files);
    const auxiliary = getAuxiliaryFiles(files);

    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: `Format: ${format.type}`,
      format: format.type,
    });

    // Parse based on format
    let modelInfo: ModelInfo;
    let config: ModelConfig | null = null;
    let tokenizerJson: unknown = null;

    if (format.type === 'gguf') {
      // Handle GGUF - delegate to existing importer logic
      modelInfo = await parseGGUFModel(format.ggufFile!, onProgress, signal);
    } else if (format.type === 'single') {
      const parsed = await parseSafetensorsFile(format.safetensorsFile!);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else if (format.type === 'sharded' || format.type === 'sharded-no-index') {
      let indexJson = null;
      if (format.indexFile) {
        indexJson = await parseIndexJson(format.indexFile);
      }
      const parsed = await parseSafetensorsSharded(format.safetensorsFiles!, indexJson);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else {
      throw new Error(`Unsupported format: ${format.type}`);
    }

    // Parse tokenizer if available
    if (auxiliary.tokenizer) {
      tokenizerJson = await parseTokenizerJson(auxiliary.tokenizer);
      modelInfo.tokenizerJson = tokenizerJson;
    }

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Determine model ID
    modelId = userModelId || extractModelId(files, config) || 'converted-model';
    modelId = sanitizeModelId(modelId);

    onProgress?.({
      stage: ConvertStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      tensorCount: modelInfo.tensors.length,
      totalSize: formatBytes(calculateTotalSize(modelInfo as SafetensorsParseResult)),
    });

    // Open model directory in OPFS
    modelDir = await openModelDirectory(modelId);

    // Process tensors and write shards
    const result = await writeTensorsToShards(modelInfo, modelDir, shardInfos, {
      quantize,
      onProgress,
      signal,
    });

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Create manifest
    onProgress?.({
      stage: ConvertStage.MANIFEST,
      message: 'Creating manifest...',
    });

    const manifest = createManifest(modelInfo, shardInfos, config, tokenizerJson);
    manifest.modelId = modelId;

    // Save manifest
    await saveManifest(JSON.stringify(manifest, null, 2));

    onProgress?.({
      stage: ConvertStage.COMPLETE,
      message: 'Conversion complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: formatBytes(result.totalSize),
    });

    return modelId;
  } catch (error) {
    // Cleanup on error
    if (modelId) {
      try {
        await deleteModel(modelId);
      } catch {
        // Ignore cleanup errors
      }
    }

    onProgress?.({
      stage: ConvertStage.ERROR,
      message: (error as Error).message,
      error: error as Error,
    });

    throw error;
  }
}

/**
 * Parse GGUF model file
 */
async function parseGGUFModel(
  file: File,
  onProgress?: (progress: ConvertProgress) => void,
  signal?: AbortSignal
): Promise<ModelInfo> {
  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing GGUF header...',
  });

  const headerBlob = file.slice(0, 10 * 1024 * 1024);
  const headerBuffer = await headerBlob.arrayBuffer();
  const ggufInfo = parseGGUFHeader(headerBuffer);

  return {
    format: 'gguf',
    tensors: ggufInfo.tensors.map((t) => ({
      ...t,
      file,
      offset: t.offset,
    })),
    config: ggufInfo.config,
    architecture: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    tensorDataOffset: ggufInfo.tensorDataOffset,
    file,
  };
}

/**
 * Extract model ID from files or config
 */
function extractModelId(files: File[], config: ModelConfig | null): string | null {
  // Try config first
  if (config?._name_or_path) {
    const parts = config._name_or_path.split('/');
    return parts[parts.length - 1];
  }

  // Try first safetensors file name
  const stFile = files.find((f) => f.name.endsWith('.safetensors'));
  if (stFile) {
    return stFile.name.replace(/\.safetensors$/, '').replace(/model[-_.]?/, '');
  }

  // Try GGUF file name
  const ggufFile = files.find((f) => f.name.endsWith('.gguf'));
  if (ggufFile) {
    return ggufFile.name.replace(/\.gguf$/, '');
  }

  return null;
}

/**
 * Write tensors to shards in OPFS
 */
async function writeTensorsToShards(
  modelInfo: ModelInfo,
  modelDir: FileSystemDirectoryHandle,
  shardInfos: ShardInfo[],
  options: {
    quantize?: string | null;
    onProgress?: (progress: ConvertProgress) => void;
    signal?: AbortSignal;
  }
): Promise<WriteResult> {
  const { quantize, onProgress, signal } = options;
  const tensors = modelInfo.tensors;
  const totalTensors = tensors.length;

  let currentShardIndex = 0;
  let currentShardData: Uint8Array[] = [];
  let currentShardSize = 0;
  let totalSize = 0;

  // Track tensor locations for manifest
  const tensorLocations: Record<string, TensorLocation> = {};

  for (let i = 0; i < tensors.length; i++) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    const tensor = tensors[i];

    onProgress?.({
      stage: ConvertStage.WRITING,
      message: `Processing ${tensor.name}`,
      current: i + 1,
      total: totalTensors,
      percent: Math.round(((i + 1) / totalTensors) * 100),
    });

    // Read tensor data
    let data = await readTensorData(tensor as TensorInfo);

    // Handle GGUF format (data is relative to tensorDataOffset)
    if (modelInfo.format === 'gguf' && modelInfo.tensorDataOffset && modelInfo.file) {
      const file = modelInfo.file;
      const blob = file.slice(tensor.offset, tensor.offset + tensor.size);
      data = await blob.arrayBuffer();
    }

    // Optionally quantize (TODO: implement Q4_K_M in browser)
    // For now, pass through as-is
    const tensorData = new Uint8Array(data);

    // Record tensor location
    const tensorStartOffset = totalSize;

    // Add to current shard, splitting if necessary
    let remaining = tensorData;
    const tensorSpans: TensorSpan[] = [];

    while (remaining.length > 0) {
      const availableInShard = SHARD_SIZE - currentShardSize;
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
      if (currentShardSize >= SHARD_SIZE) {
        await flushShard(modelDir, currentShardIndex, currentShardData, shardInfos);
        currentShardIndex++;
        currentShardData = [];
        currentShardSize = 0;
      }
    }

    // Record tensor location in manifest format
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
  }

  // Flush final shard
  if (currentShardData.length > 0) {
    await flushShard(modelDir, currentShardIndex, currentShardData, shardInfos);
  }

  return { totalSize, tensorLocations };
}

/**
 * Flush shard data to OPFS
 */
async function flushShard(
  modelDir: FileSystemDirectoryHandle,
  shardIndex: number,
  dataChunks: Uint8Array[],
  shardInfos: ShardInfo[]
): Promise<void> {
  // Concatenate chunks
  const totalSize = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const shardData = new Uint8Array(totalSize);
  let offset = 0;
  for (const chunk of dataChunks) {
    shardData.set(chunk, offset);
    offset += chunk.length;
  }

  // Compute hash
  const hash = await computeSHA256(shardData);

  // Write to OPFS
  const filename = generateShardFilename(shardIndex);
  const fileHandle = await modelDir.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(shardData);
  await writable.close();

  shardInfos.push({
    index: shardIndex,
    filename,
    size: shardData.length,
    hash,
    offset: shardIndex * SHARD_SIZE,
  });
}

/**
 * Create RDRR manifest
 */
function createManifest(
  modelInfo: ModelInfo,
  shardInfos: ShardInfo[],
  config: ModelConfig | null,
  tokenizerJson: unknown
): RDRRManifest {
  const architecture = extractArchitecture(modelInfo, config);

  const manifest: RDRRManifest = {
    version: RDRR_VERSION,
    modelId: 'converted-model',
    modelType: config?.architectures?.[0] || modelInfo.architecture || 'unknown',
    quantization: modelInfo.quantization || 'F16',
    architecture,
    shards: shardInfos,
    tensors: buildTensorMap(modelInfo.tensors, shardInfos),
    totalSize: shardInfos.reduce((sum, s) => sum + s.size, 0),
    hashAlgorithm: 'sha256',
    metadata: {
      source: 'browser-converter',
      convertedAt: new Date().toISOString(),
    },
  };

  // Include tokenizer if available
  if (tokenizerJson) {
    const tokenizer = tokenizerJson as { model?: { vocab?: unknown[] | Record<string, unknown> } };
    manifest.tokenizer = {
      type: 'bundled',
      vocabSize:
        (tokenizer.model?.vocab as unknown[])?.length ||
        Object.keys((tokenizer.model?.vocab as Record<string, unknown>) || {}).length,
    };
    // Store tokenizer separately or embed
    manifest.metadata.hasTokenizer = true;
  }

  return manifest;
}

/**
 * Extract architecture info from model
 */
function extractArchitecture(
  modelInfo: ModelInfo,
  config: ModelConfig | null
): ArchitectureConfig {
  if (config) {
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
  if (modelInfo.config) {
    const c = modelInfo.config as GGUFModelConfig;
    return {
      numLayers: (c.blockCount ?? 32) as number,
      hiddenSize: (c.embeddingLength ?? 4096) as number,
      intermediateSize: (c.feedForwardLength ?? 11008) as number,
      numAttentionHeads: (c.attentionHeadCount ?? 32) as number,
      numKeyValueHeads: (c.attentionHeadCountKV ?? 32) as number,
      headDim: Math.floor(((c.embeddingLength ?? 4096) as number) / ((c.attentionHeadCount ?? 32) as number)),
      vocabSize: (c.vocabSize ?? 32000) as number,
      maxSeqLen: (c.contextLength ?? 2048) as number,
    };
  }

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
function buildTensorMap(
  tensors: InternalTensorInfo[],
  shardInfos: ShardInfo[]
): Record<string, TensorLocation> {
  const tensorMap: Record<string, TensorLocation> = {};

  let globalOffset = 0;
  for (const tensor of tensors) {
    const startShard = Math.floor(globalOffset / SHARD_SIZE);
    const offsetInShard = globalOffset % SHARD_SIZE;

    if (offsetInShard + tensor.size <= SHARD_SIZE) {
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
        const available = SHARD_SIZE - currentOffset;
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
 * Check if conversion is supported in this browser
 */
export function isConversionSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    'storage' in navigator &&
    'getDirectory' in (navigator.storage as unknown as { getDirectory?: unknown })
  );
}

/**
 * Pick model files using File System Access API
 * @returns Selected files
 */
export async function pickModelFiles(): Promise<File[]> {
  // Try directory picker first (for HuggingFace models)
  if ('showDirectoryPicker' in window) {
    try {
      const dirHandle = await (window as Window & { showDirectoryPicker: (opts?: { mode?: string }) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker({
        mode: 'read',
      });
      return await collectFilesFromDirectory(dirHandle);
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      // Fall back to file picker
    }
  }

  // Fall back to file picker
  if ('showOpenFilePicker' in window) {
    const handles = await (window as Window & {
      showOpenFilePicker: (opts?: {
        multiple?: boolean;
        types?: Array<{ description: string; accept: Record<string, string[]> }>;
      }) => Promise<FileSystemFileHandle[]>;
    }).showOpenFilePicker({
      multiple: true,
      types: [
        {
          description: 'Model files',
          accept: {
            'application/octet-stream': ['.gguf', '.safetensors', '.bin'],
            'application/json': ['.json'],
          },
        },
      ],
    });
    return Promise.all(handles.map((h) => h.getFile()));
  }

  // Ultimate fallback: input element
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.gguf,.safetensors,.json,.bin';
    input.onchange = () => {
      resolve(Array.from(input.files || []));
    };
    input.click();
  });
}

/**
 * Collect all files from a directory handle recursively
 */
async function collectFilesFromDirectory(
  dirHandle: FileSystemDirectoryHandle,
  files: File[] = []
): Promise<File[]> {
  // Use type assertion for directory iteration - handles may have values() or be async iterable
  const entries = (dirHandle as unknown as { values(): AsyncIterable<FileSystemHandle> }).values();
  for await (const entry of entries) {
    if (entry.kind === 'file') {
      const file = await (entry as FileSystemFileHandle).getFile();
      // Only include relevant files
      if (
        file.name.endsWith('.safetensors') ||
        file.name.endsWith('.gguf') ||
        file.name.endsWith('.json') ||
        file.name === 'tokenizer.model'
      ) {
        files.push(file);
      }
    } else if (entry.kind === 'directory') {
      // Don't recurse into subdirectories for model files
      // HuggingFace models are flat
    }
  }
  return files;
}

export default convertModel;
