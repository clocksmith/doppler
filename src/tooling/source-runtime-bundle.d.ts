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
    scaleSemantics: 'step' | 'qmax_abs';
    scaleDivisor?: number;
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
    scaleSemantics: 'step' | 'qmax_abs';
    scaleDivisor?: number;
    storageShape: [number, number];
    quantAxis: 0 | 1;
    scaleCompanionDtype?: 'UINT8';
    scaleCompanionDequant?: {
      scale: number;
      zeroPoint: number;
    };
    scaleSourcePath: string;
    scaleOffset: number;
    scaleSize: number;
    sumSourcePath?: string;
    sumOffset?: number;
    sumSize?: number;
  } | {
    kind: 'litert_axis_blocked_dequant';
    scheme: 'per_axis_affine';
    sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
    targetDtype: 'F16';
    storageEncoding: 'signed' | 'offset_binary';
    scaleSemantics: 'step' | 'qmax_abs';
    scaleDivisor?: number;
    storageShape: [number, number];
    quantAxis: 0;
    storageBlockSize: number;
    storageLaneOrder: number[];
    scaleSourcePath: string;
    scaleOffset: number;
    scaleSize: number;
    sumSourcePath?: string;
    sumOffset?: number;
    sumSize?: number;
  } | null;
}

export type { SourceRuntimeFile, SourceRuntimeShardSource, SourceRuntimeTokenizerMetadata, SourceRuntimeMetadata } from '../formats/source-runtime.js';
import type { SourceRuntimeFile, SourceRuntimeShardSource, SourceRuntimeMetadata } from '../formats/source-runtime.js';
export { createSourceStorageContext } from '../storage/source-storage-context.js';
export type { CreateSourceStorageContextOptions, SourceStorageContext } from '../storage/source-storage-context.js';

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
