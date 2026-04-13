import type { TensorRole, TensorSourceTransform } from '../formats/rdrr/index.js';
import type { ManifestEmbeddingPostprocessorSchema } from '../config/schema/index.js';
import type { RuntimeModelContract } from '../inference/runtime-model.js';
import type { SourceArtifactKind } from './source-artifact-adapter.js';

export declare const DIRECT_SOURCE_RUNTIME_MODE: 'direct-source';
export declare const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION: 1;
export declare const DIRECT_SOURCE_RUNTIME_SCHEMA: 'direct-source/v1';
export declare const DIRECT_SOURCE_PATH_RUNTIME_LOCAL: 'runtime-local';
export declare const DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE: 'artifact-relative';

export interface SourceRuntimeTensor {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  offset: number;
  sourcePath: string;
  layout?: string | null;
  role?: TensorRole;
  group?: string | null;
  sourceTransform?: TensorSourceTransform | {
    kind: 'litert_rowwise_dequant';
    scheme: 'per_row_affine';
    sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
    targetDtype: 'F16';
    storageEncoding: 'signed' | 'offset_binary';
    scaleSourcePath: string;
    scaleOffset: number;
    scaleSize: number;
    rowSumSourcePath?: string;
    rowSumOffset?: number;
    rowSumSize?: number;
  } | {
    kind: 'litert_axis_dequant';
    scheme: 'per_axis_affine';
    sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
    targetDtype: 'F16';
    storageEncoding: 'signed' | 'offset_binary';
    storageShape: [number, number];
    quantAxis: 0 | 1;
    scaleSourcePath: string;
    scaleOffset: number;
    scaleSize: number;
    sumSourcePath?: string;
    sumOffset?: number;
    sumSize?: number;
  } | null;
}

export interface SourceRuntimeFile {
  path: string;
  size: number;
  hash?: string | null;
  hashAlgorithm?: string | null;
  kind?: string | null;
}

export interface SourceRuntimeShardSource {
  index: number;
  path: string;
  filename: string;
  size: number;
  hash: string;
  hashAlgorithm: string;
}

export interface SourceRuntimeTokenizerMetadata {
  jsonPath: string | null;
  configPath: string | null;
  modelPath: string | null;
}

export interface SourceRuntimeMetadata {
  mode: 'direct-source';
  schema: 'direct-source/v1';
  schemaVersion: 1;
  sourceKind: SourceArtifactKind | 'rdrr' | null;
  hashAlgorithm: string;
  pathSemantics: 'runtime-local' | 'artifact-relative';
  sourceFiles: SourceRuntimeShardSource[];
  auxiliaryFiles: SourceRuntimeFile[];
  tokenizer: SourceRuntimeTokenizerMetadata;
}

export interface BuildSourceRuntimeBundleOptions {
  modelId: string;
  modelName?: string | null;
  modelType: string;
  sourceKind?: SourceArtifactKind | 'rdrr' | null;
  architecture: Record<string, unknown> | string | null;
  architectureHint?: string | null;
  rawConfig?: Record<string, unknown> | null;
  manifestConfig?: {
    visionConfig?: Record<string, unknown> | null;
    audioConfig?: Record<string, unknown> | null;
  } | null;
  inference: Record<string, unknown>;
  tensors: SourceRuntimeTensor[];
  embeddingPostprocessor?: ManifestEmbeddingPostprocessorSchema | null;
  sourceFiles?: SourceRuntimeFile[] | null;
  auxiliaryFiles?: SourceRuntimeFile[] | null;
  resolveSourceSize?: ((path: string) => Promise<number> | number) | null;
  sourceQuantization?: string | null;
  quantizationInfo?: Record<string, unknown> | null;
  manifestQuantization?: string | null;
  hashAlgorithm?: string | null;
  tokenizerJson?: Record<string, unknown> | null;
  tokenizerConfig?: Record<string, unknown> | null;
  tokenizerModelName?: string | null;
  tokenizerJsonPath?: string | null;
  tokenizerConfigPath?: string | null;
  tokenizerModelPath?: string | null;
  eosTokenId?: number | number[] | null;
  convertedAt?: string | null;
  conversionInfo?: Record<string, unknown> | null;
}

export interface BuildSourceRuntimeBundleResult {
  model: RuntimeModelContract;
  manifest: RuntimeModelContract;
  shardSources: SourceRuntimeShardSource[];
}

export declare function buildSourceRuntimeBundle(
  options: BuildSourceRuntimeBundleOptions
): Promise<BuildSourceRuntimeBundleResult>;

export declare function getSourceRuntimeMetadata(
  manifest: RuntimeModelContract | Record<string, unknown> | null | undefined
): SourceRuntimeMetadata | null;

export interface CreateSourceStorageContextOptions {
  model?: RuntimeModelContract | null;
  manifest?: RuntimeModelContract | null;
  shardSources?: SourceRuntimeShardSource[] | null;
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
  tokenizerConfigPath?: string | null;
  tokenizerModelPath?: string | null;
  verifyHashes?: boolean;
  sourceHashesTrusted?: boolean;
}

export interface SourceStorageContext {
  loadShard: (index: number) => Promise<ArrayBuffer | Uint8Array>;
  loadShardRange: ((
    index: number,
    offset?: number,
    length?: number | null
  ) => Promise<ArrayBuffer | Uint8Array>) | null;
  streamShardRange: ((
    index: number,
    offset?: number,
    length?: number | null,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<Uint8Array>) | null;
  loadTokenizerJson: (() => Promise<Record<string, unknown> | null>) | null;
  loadTokenizerModel: ((pathHint?: string) => Promise<ArrayBuffer | null>) | null;
  verifyHashes: boolean;
}

export declare function createSourceStorageContext(
  options: CreateSourceStorageContextOptions
): SourceStorageContext;
