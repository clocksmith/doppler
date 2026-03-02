import type { RDRRManifest } from '../storage/rdrr-format.js';

export declare const DIRECT_SOURCE_RUNTIME_MODE: 'direct-source';
export declare const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION: 1;
export declare const DIRECT_SOURCE_RUNTIME_SCHEMA: 'direct-source/v1';

export interface SourceRuntimeTensor {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  offset: number;
  sourcePath: string;
  layout?: string | null;
}

export interface SourceRuntimeFile {
  path: string;
  size: number;
}

export interface SourceRuntimeShardSource {
  index: number;
  path: string;
  filename: string;
  size: number;
}

export interface BuildSourceRuntimeBundleOptions {
  modelId: string;
  modelName?: string | null;
  modelType: string;
  architecture: Record<string, unknown> | string | null;
  architectureHint?: string | null;
  rawConfig?: Record<string, unknown> | null;
  inference: Record<string, unknown>;
  tensors: SourceRuntimeTensor[];
  sourceFiles?: SourceRuntimeFile[] | null;
  resolveSourceSize?: ((path: string) => Promise<number> | number) | null;
  sourceQuantization?: string | null;
  quantizationInfo?: Record<string, unknown> | null;
  manifestQuantization?: string | null;
  hashAlgorithm?: string | null;
  tokenizerJson?: Record<string, unknown> | null;
  tokenizerConfig?: Record<string, unknown> | null;
  tokenizerModelName?: string | null;
  eosTokenId?: number | number[] | null;
  convertedAt?: string | null;
  conversionInfo?: Record<string, unknown> | null;
}

export interface BuildSourceRuntimeBundleResult {
  manifest: RDRRManifest;
  shardSources: SourceRuntimeShardSource[];
}

export declare function buildSourceRuntimeBundle(
  options: BuildSourceRuntimeBundleOptions
): Promise<BuildSourceRuntimeBundleResult>;

export interface CreateSourceStorageContextOptions {
  manifest: RDRRManifest;
  shardSources: SourceRuntimeShardSource[];
  readRange: (
    path: string,
    offset: number,
    length: number
  ) => Promise<ArrayBuffer | Uint8Array>;
  streamRange?: (
    path: string,
    offset: number,
    length: number,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<ArrayBuffer | Uint8Array>;
  readText?: (path: string) => Promise<string | Record<string, unknown> | null | undefined>;
  readBinary?: (path: string) => Promise<ArrayBuffer | Uint8Array>;
  tokenizerJsonPath?: string | null;
  tokenizerModelPath?: string | null;
  verifyHashes?: boolean;
}

export interface SourceStorageContext {
  loadShard: (index: number) => Promise<ArrayBuffer | Uint8Array>;
  loadShardRange: (
    index: number,
    offset?: number,
    length?: number | null
  ) => Promise<ArrayBuffer | Uint8Array>;
  streamShardRange: (
    index: number,
    offset?: number,
    length?: number | null,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<Uint8Array>;
  loadTokenizerJson: (() => Promise<Record<string, unknown> | null>) | null;
  loadTokenizerModel: ((pathHint?: string) => Promise<ArrayBuffer | null>) | null;
  verifyHashes: boolean;
}

export declare function createSourceStorageContext(
  options: CreateSourceStorageContextOptions
): SourceStorageContext;
