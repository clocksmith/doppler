/**
 * Storage and Shard Types
 */

import type { DataType, TensorMetadata } from './tensor.js';
import type { ModelConfig, QuantizationConfig } from './model.js';

/** RDRR manifest (on-disk schema) */
export interface RdrrManifest {
  version: number;
  modelId: string;
  architecture: string;
  config: ModelConfig;
  quantization?: QuantizationConfig;
  tensors: TensorMetadata[];
  shards: ShardInfo[];
  tokenizer?: TokenizerInfo;
  metadata?: Record<string, unknown>;
}

/** Shard information */
export interface ShardInfo {
  index: number;
  fileName: string;
  byteLength: number;
  checksum?: string;
  tensors: string[];
}

/** Tokenizer information in manifest */
export interface TokenizerInfo {
  type: 'sentencepiece' | 'bpe' | 'tiktoken';
  vocabFile?: string;
  mergesFile?: string;
  addedTokens?: Record<string, number>;
}

/** Normalized manifest (runtime) */
export interface NormalizedManifest {
  modelId: string;
  architecture: string;
  config: ModelConfig;
  tensors: Map<string, TensorMetadata>;
  shards: ShardInfo[];
  tensorsByName: Map<string, TensorMetadata>;
  tensorsByShard: Map<number, TensorMetadata[]>;
}

/** Shard resolver interface */
export interface ShardResolver {
  /** Resolve a shard by index */
  resolve(index: number): Promise<Uint8Array>;

  /** Check if shard is available */
  has(index: number): Promise<boolean>;

  /** Get shard info */
  info(index: number): ShardInfo | undefined;

  /** Prefetch shards */
  prefetch?(indices: number[]): Promise<void>;
}

/** Shard source types */
export type ShardSourceType = 'opfs' | 'http' | 'bridge' | 'p2p' | 'memory';

/** Shard source interface */
export interface ShardSource {
  readonly type: ShardSourceType;
  readonly priority: number;

  /** Check if source can provide shard */
  canProvide(index: number): Promise<boolean>;

  /** Load shard data */
  load(index: number, info: ShardInfo): Promise<Uint8Array>;

  /** Get loading progress */
  getProgress?(index: number): number;
}

/** HTTP shard source options */
export interface HttpSourceOptions {
  baseUrl: string;
  retryCount?: number;
  retryDelay?: number;
  timeout?: number;
  headers?: Record<string, string>;
}

/** OPFS shard store options */
export interface OpfsStoreOptions {
  modelId: string;
  createIfMissing?: boolean;
}

/** Download progress */
export interface DownloadProgress {
  shardIndex: number;
  bytesLoaded: number;
  bytesTotal: number;
  percent: number;
  speed?: number;
}

/** Download options */
export interface DownloadOptions {
  /** Base URL for model files */
  baseUrl: string;
  /** Progress callback */
  onProgress?: (progress: DownloadProgress) => void;
  /** Abort signal */
  signal?: AbortSignal;
  /** Number of concurrent downloads */
  concurrency?: number;
  /** Retry configuration */
  retry?: RetryConfig;
}

/** Retry configuration */
export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffFactor: number;
}

/** Storage quota information */
export interface QuotaInfo {
  usage: number;
  quota: number;
  available: number;
  percentUsed: number;
}

/** Model storage entry */
export interface ModelStorageEntry {
  modelId: string;
  architecture: string;
  size: number;
  shardCount: number;
  downloadedAt: Date;
  lastUsedAt: Date;
  manifest: RdrrManifest;
}

/** Storage manager interface */
export interface StorageManager {
  /** List stored models */
  list(): Promise<ModelStorageEntry[]>;

  /** Check if model is stored */
  has(modelId: string): Promise<boolean>;

  /** Get model manifest */
  getManifest(modelId: string): Promise<RdrrManifest | null>;

  /** Delete model */
  delete(modelId: string): Promise<void>;

  /** Get quota info */
  getQuota(): Promise<QuotaInfo>;

  /** Create shard resolver for model */
  createResolver(modelId: string): Promise<ShardResolver>;
}
