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

/**
 * Progress stages
 */
export declare const ImportStage: {
  readonly PARSING: 'parsing';
  readonly SHARDING: 'sharding';
  readonly WRITING: 'writing';
  readonly COMPLETE: 'complete';
  readonly ERROR: 'error';
};

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
 * Import a GGUF file to OPFS as .rdrr format
 *
 * @param file - GGUF file to import
 * @param options - Import options
 * @returns Model ID
 */
export declare function importGGUFFile(
  file: File,
  options?: ImportOptions
): Promise<string>;

export default importGGUFFile;
