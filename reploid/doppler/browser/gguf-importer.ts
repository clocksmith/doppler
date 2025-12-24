/**
 * gguf-importer.ts - Stream GGUF to .rdrr shards in OPFS
 *
 * Handles:
 * - Streaming large GGUF files without loading into memory
 * - Chunking into 64MB shards
 * - SHA-256 hash per shard (BLAKE3 fallback if available)
 * - Writing to OPFS via shard-manager
 * - Progress reporting
 * - Abort/cancel support
 *
 * @module browser/gguf-importer
 */

import { parseGGUFHeader, GGUFParseResult, GGUFTensor } from './gguf-parser-browser.js';
import { canStreamFile } from './file-picker.js';
import {
  initOPFS,
  openModelDirectory,
  saveManifest,
  deleteModel,
} from '../storage/shard-manager.js';
import {
  SHARD_SIZE,
  RDRR_VERSION,
  generateShardFilename,
} from '../storage/rdrr-format.js';

// Header size to read for parsing (10MB should cover any GGUF header)
const HEADER_READ_SIZE = 10 * 1024 * 1024;

// ============================================================================
// Types
// ============================================================================

/**
 * Progress stages
 */
export const ImportStage = {
  PARSING: 'parsing',
  SHARDING: 'sharding',
  WRITING: 'writing',
  COMPLETE: 'complete',
  ERROR: 'error',
} as const;

export type ImportStageType = (typeof ImportStage)[keyof typeof ImportStage];

/**
 * Progress callback payload
 */
export interface ImportProgress {
  stage: ImportStageType;
  message: string;
  filename?: string;
  modelId?: string;
  architecture?: string;
  quantization?: string;
  current?: number;
  total?: number;
  percent?: number;
  shardCount?: number;
  totalSize?: number;
  error?: Error;
}

/**
 * Import options
 */
export interface ImportOptions {
  onProgress?: (progress: ImportProgress) => void;
  signal?: AbortSignal;
}

/**
 * Shard info stored in manifest
 */
export interface ShardInfo {
  index: number;
  filename: string;
  size: number;
  hash: string;
  offset: number;
}

/**
 * Tensor location in manifest (single shard)
 */
export interface TensorLocationSingle {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
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
 * Tensor location in manifest (multi-shard)
 */
export interface TensorLocationMulti {
  spans: TensorSpan[];
  size: number;
  shape: number[];
  dtype: string;
}

export type TensorLocation = TensorLocationSingle | TensorLocationMulti;

/**
 * MoE configuration
 */
export interface MoEConfig {
  numExperts: number;
  numExpertsPerToken: number;
  expertSize: number;
  expertShardMap: unknown[];
}

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
  moeConfig: MoEConfig | null;
  shards: ShardInfo[];
  tensors: Record<string, TensorLocation>;
  totalSize: number;
  fullHash: string;
  hashAlgorithm: string;
  metadata: {
    source: string;
    originalFile: string;
    importedAt: string;
    ggufVersion: number;
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute SHA-256 hash of data
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
 * Sanitize model ID for filesystem
 */
function sanitizeModelId(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9_-]/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 64) || 'imported-model'
  );
}

// ============================================================================
// Main Import Function
// ============================================================================

/**
 * Import a GGUF file to OPFS as .rdrr format
 *
 * @param file - GGUF file to import
 * @param options - Import options
 * @returns Model ID
 */
export async function importGGUFFile(
  file: File,
  { onProgress, signal }: ImportOptions = {}
): Promise<string> {
  let modelId: string | null = null;
  let modelDir: FileSystemDirectoryHandle | null = null;
  const shardInfos: ShardInfo[] = [];

  try {
    // Initialize OPFS
    await initOPFS();

    // Report parsing stage
    onProgress?.({
      stage: ImportStage.PARSING,
      message: 'Parsing GGUF header...',
      filename: file.name,
    });

    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Read header portion for parsing
    const headerBlob = file.slice(0, Math.min(HEADER_READ_SIZE, file.size));
    const headerBuffer = await headerBlob.arrayBuffer();
    const ggufInfo = parseGGUFHeader(headerBuffer);

    // Generate model ID from filename or GGUF metadata
    modelId = sanitizeModelId(
      ggufInfo.modelName !== 'unknown' ? ggufInfo.modelName : file.name.replace(/\.gguf$/i, '')
    );

    onProgress?.({
      stage: ImportStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      architecture: ggufInfo.architecture,
      quantization: ggufInfo.quantization,
    });

    // Open model directory in OPFS
    modelDir = await openModelDirectory(modelId);

    // Calculate expected shard count
    const totalDataSize = file.size - ggufInfo.tensorDataOffset;
    const expectedShards = Math.ceil(totalDataSize / SHARD_SIZE);

    onProgress?.({
      stage: ImportStage.SHARDING,
      message: `Preparing ${expectedShards} shards...`,
      current: 0,
      total: expectedShards,
      percent: 0,
    });

    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Stream the file and create shards
    if (canStreamFile(file)) {
      await streamToShards(file, ggufInfo, modelDir, shardInfos, {
        onProgress,
        signal,
      });
    } else {
      // Fallback for browsers without streaming
      await bufferToShards(file, ggufInfo, modelDir, shardInfos, {
        onProgress,
        signal,
      });
    }

    // Check for abort before finalizing
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Create manifest
    const manifest = createManifest(ggufInfo, shardInfos, file.size);

    onProgress?.({
      stage: ImportStage.WRITING,
      message: 'Saving manifest...',
    });

    // Save manifest to OPFS
    await saveManifest(JSON.stringify(manifest, null, 2));

    onProgress?.({
      stage: ImportStage.COMPLETE,
      message: 'Import complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: file.size,
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
      stage: ImportStage.ERROR,
      message: (error as Error).message,
      error: error as Error,
    });

    throw error;
  }
}

/**
 * Stream file to shards using ReadableStream
 */
async function streamToShards(
  file: File,
  ggufInfo: GGUFParseResult,
  modelDir: FileSystemDirectoryHandle,
  shardInfos: ShardInfo[],
  { onProgress, signal }: ImportOptions
): Promise<void> {
  const tensorDataOffset = ggufInfo.tensorDataOffset;
  const totalDataSize = file.size - tensorDataOffset;
  const expectedShards = Math.ceil(totalDataSize / SHARD_SIZE);

  // Slice to just tensor data
  const tensorBlob = file.slice(tensorDataOffset);
  const stream = tensorBlob.stream();
  const reader = stream.getReader();

  let shardIndex = 0;
  let shardBuffer = new Uint8Array(SHARD_SIZE);
  let shardOffset = 0;
  let totalProcessed = 0;

  try {
    while (true) {
      // Check for abort
      if (signal?.aborted) {
        reader.cancel();
        throw new DOMException('Import cancelled', 'AbortError');
      }

      const { done, value } = await reader.read();

      if (done) {
        // Write final partial shard if any data remains
        if (shardOffset > 0) {
          await writeShard(modelDir, shardIndex, shardBuffer.slice(0, shardOffset), shardInfos);
          shardIndex++;
        }
        break;
      }

      // Process chunk
      let chunkOffset = 0;
      while (chunkOffset < value.length) {
        const remaining = SHARD_SIZE - shardOffset;
        const toCopy = Math.min(remaining, value.length - chunkOffset);

        shardBuffer.set(value.subarray(chunkOffset, chunkOffset + toCopy), shardOffset);
        shardOffset += toCopy;
        chunkOffset += toCopy;
        totalProcessed += toCopy;

        // Shard full, write it
        if (shardOffset === SHARD_SIZE) {
          await writeShard(modelDir, shardIndex, shardBuffer, shardInfos);

          shardIndex++;
          shardBuffer = new Uint8Array(SHARD_SIZE);
          shardOffset = 0;

          onProgress?.({
            stage: ImportStage.SHARDING,
            message: `Writing shard ${shardIndex}/${expectedShards}`,
            current: shardIndex,
            total: expectedShards,
            percent: Math.round((totalProcessed / totalDataSize) * 100),
          });
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Buffer-based fallback for browsers without streaming
 */
async function bufferToShards(
  file: File,
  ggufInfo: GGUFParseResult,
  modelDir: FileSystemDirectoryHandle,
  shardInfos: ShardInfo[],
  { onProgress, signal }: ImportOptions
): Promise<void> {
  const tensorDataOffset = ggufInfo.tensorDataOffset;
  const totalDataSize = file.size - tensorDataOffset;
  const expectedShards = Math.ceil(totalDataSize / SHARD_SIZE);

  console.warn('[GGUF Import] Using buffer fallback - large files may cause memory issues');

  let shardIndex = 0;
  let offset = tensorDataOffset;

  while (offset < file.size) {
    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    const end = Math.min(offset + SHARD_SIZE, file.size);
    const blob = file.slice(offset, end);
    const buffer = await blob.arrayBuffer();
    const data = new Uint8Array(buffer);

    await writeShard(modelDir, shardIndex, data, shardInfos);

    shardIndex++;
    offset = end;

    onProgress?.({
      stage: ImportStage.SHARDING,
      message: `Writing shard ${shardIndex}/${expectedShards}`,
      current: shardIndex,
      total: expectedShards,
      percent: Math.round(((offset - tensorDataOffset) / totalDataSize) * 100),
    });
  }
}

/**
 * Write a shard to OPFS
 */
async function writeShard(
  modelDir: FileSystemDirectoryHandle,
  shardIndex: number,
  data: Uint8Array,
  shardInfos: ShardInfo[]
): Promise<void> {
  const filename = generateShardFilename(shardIndex);
  const hash = await computeSHA256(data);

  // Get file handle and write
  const fileHandle = await modelDir.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(new Uint8Array(data.buffer as ArrayBuffer, data.byteOffset, data.length));
  await writable.close();

  // Record shard info
  shardInfos.push({
    index: shardIndex,
    filename,
    size: data.length,
    hash: hash, // SHA-256 hash
    offset: shardIndex * SHARD_SIZE,
  });
}

/**
 * Create .rdrr manifest from GGUF info and shards
 */
function createManifest(
  ggufInfo: GGUFParseResult,
  shardInfos: ShardInfo[],
  fileSize: number
): RDRRManifest {
  const config = ggufInfo.config;

  // Build architecture object
  const architecture: ArchitectureConfig = {
    numLayers: config.blockCount || 32,
    hiddenSize: config.embeddingLength || 4096,
    intermediateSize: config.feedForwardLength || 11008,
    numAttentionHeads: config.attentionHeadCount || 32,
    numKeyValueHeads: config.attentionHeadCountKV || config.attentionHeadCount || 32,
    headDim: config.embeddingLength
      ? Math.floor(config.embeddingLength / (config.attentionHeadCount || 32))
      : 128,
    vocabSize: config.vocabSize || 32000,
    maxSeqLen: config.contextLength || 2048,
  };

  // Build MoE config if applicable
  let moeConfig: MoEConfig | null = null;
  if (config.expertCount) {
    moeConfig = {
      numExperts: config.expertCount,
      numExpertsPerToken: config.expertUsedCount || 2,
      expertSize: 0, // Would need to calculate from tensors
      expertShardMap: [],
    };
  }

  // Calculate total size from shards
  const totalSize = shardInfos.reduce((sum, s) => sum + s.size, 0);

  // Compute full file hash placeholder (would need full hash for real impl)
  const fullHash = shardInfos.length > 0 ? shardInfos[0].hash : '';

  // Build tensor location map
  // Maps each tensor to its shard(s) and offset within shard
  const tensors = buildTensorLocations(ggufInfo.tensors, ggufInfo.tensorDataOffset);

  return {
    version: RDRR_VERSION,
    modelId:
      ggufInfo.modelName !== 'unknown' ? sanitizeModelId(ggufInfo.modelName) : 'imported-model',
    modelType: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    architecture,
    moeConfig,
    shards: shardInfos,
    tensors,
    totalSize,
    fullHash,
    hashAlgorithm: 'sha256',
    metadata: {
      source: 'browser-import',
      originalFile: ggufInfo.modelName,
      importedAt: new Date().toISOString(),
      ggufVersion: ggufInfo.version,
    },
  };
}

/**
 * Build tensor location map from GGUF tensor info
 * Maps absolute file offsets to shard-relative positions
 *
 * @param ggufTensors - Tensors from GGUF parser
 * @param tensorDataOffset - Offset where tensor data starts in GGUF
 * @returns Map of tensor name -> location info
 */
function buildTensorLocations(
  ggufTensors: GGUFTensor[],
  tensorDataOffset: number
): Record<string, TensorLocation> {
  const tensors: Record<string, TensorLocation> = {};

  for (const tensor of ggufTensors) {
    // Position relative to tensor data start (not file start)
    const relativeOffset = tensor.offset - tensorDataOffset;

    // Which shard does this tensor start in?
    const startShard = Math.floor(relativeOffset / SHARD_SIZE);
    const offsetInShard = relativeOffset % SHARD_SIZE;

    // Does tensor fit entirely in one shard?
    const endOffset = offsetInShard + tensor.size;

    if (endOffset <= SHARD_SIZE) {
      // Tensor fits in single shard
      tensors[tensor.name] = {
        shard: startShard,
        offset: offsetInShard,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    } else {
      // Tensor spans multiple shards - create spans array
      const spans: TensorSpan[] = [];
      let remaining = tensor.size;
      let currentShard = startShard;
      let currentOffset = offsetInShard;

      while (remaining > 0) {
        const availableInShard = SHARD_SIZE - currentOffset;
        const chunkSize = Math.min(remaining, availableInShard);

        spans.push({
          shardIndex: currentShard,
          offset: currentOffset,
          size: chunkSize,
        });

        remaining -= chunkSize;
        currentShard++;
        currentOffset = 0; // Next shard starts at offset 0
      }

      tensors[tensor.name] = {
        spans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    }
  }

  return tensors;
}

/**
 * Check if GGUF import is supported in this browser
 */
export function isImportSupported(): boolean {
  // Need OPFS and either streaming or array buffer
  return (
    typeof navigator !== 'undefined' &&
    'storage' in navigator &&
    'getDirectory' in (navigator.storage as unknown as { getDirectory?: unknown })
  );
}

export default importGGUFFile;
