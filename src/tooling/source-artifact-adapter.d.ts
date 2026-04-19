import type { ManifestEmbeddingPostprocessorSchema } from '../config/schema/index.js';
import type { RuntimeModelContract } from '../inference/runtime-model.js';
import type {
  BuildSourceRuntimeBundleOptions,
  BuildSourceRuntimeBundleResult,
  SourceRuntimeFile,
  SourceRuntimeTensor,
} from './source-runtime-bundle.js';

export declare const SOURCE_ARTIFACT_KIND_SAFETENSORS: 'safetensors';
export declare const SOURCE_ARTIFACT_KIND_GGUF: 'gguf';
export declare const SOURCE_ARTIFACT_KIND_TFLITE: 'tflite';
export declare const SOURCE_ARTIFACT_KIND_LITERT_TASK: 'litert-task';
export declare const SOURCE_ARTIFACT_KIND_LITERTLM: 'litertlm';

export type SourceArtifactKind =
  | typeof SOURCE_ARTIFACT_KIND_SAFETENSORS
  | typeof SOURCE_ARTIFACT_KIND_GGUF
  | typeof SOURCE_ARTIFACT_KIND_TFLITE
  | typeof SOURCE_ARTIFACT_KIND_LITERT_TASK
  | typeof SOURCE_ARTIFACT_KIND_LITERTLM;

export type DirectSourceRuntimeKind =
  | typeof SOURCE_ARTIFACT_KIND_SAFETENSORS
  | typeof SOURCE_ARTIFACT_KIND_GGUF
  | typeof SOURCE_ARTIFACT_KIND_TFLITE
  | typeof SOURCE_ARTIFACT_KIND_LITERT_TASK
  | typeof SOURCE_ARTIFACT_KIND_LITERTLM;

export interface ParsedSourceArtifact {
  sourceKind: SourceArtifactKind | string;
  config: Record<string, unknown>;
  tensors: SourceRuntimeTensor[];
  architectureHint?: string | null;
  embeddingPostprocessor?: ManifestEmbeddingPostprocessorSchema | null;
  modelType?: string | null;
  architecture: Record<string, unknown> | string | null;
  manifestConfig?: Record<string, unknown> | null;
  manifestInference?: Record<string, unknown> | null;
  sourceQuantization?: string | null;
  tokenizerJson?: Record<string, unknown> | null;
  tokenizerConfig?: Record<string, unknown> | null;
  tokenizerModelName?: string | null;
  tokenizerJsonPath?: string | null;
  tokenizerConfigPath?: string | null;
  tokenizerModelPath?: string | null;
  sourceFiles: SourceRuntimeFile[];
  auxiliaryFiles: SourceRuntimeFile[];
  sourcePathForModelId?: string | null;
}

export interface ResolveSourceRuntimeBundleFromParsedArtifactOptions {
  parsedArtifact: ParsedSourceArtifact;
  requestedModelId?: string | null;
  modelKind?: string | null;
  runtimeLabel?: string | null;
  logCategory?: string | null;
  hashFileEntries: (
    entries: SourceRuntimeFile[] | null | undefined,
    hashAlgorithm: string
  ) => Promise<SourceRuntimeFile[]>;
}

export interface ResolvedSourceRuntimeArtifactBundle extends BuildSourceRuntimeBundleResult {
  model: RuntimeModelContract;
  manifest: RuntimeModelContract;
  sourceKind: DirectSourceRuntimeKind;
  sourceQuantization: string;
  sourceFiles: SourceRuntimeFile[];
  auxiliaryFiles: SourceRuntimeFile[];
  hashAlgorithm: string;
  modelId: string;
  plan: {
    modelType: string;
    manifestInference: BuildSourceRuntimeBundleOptions['inference'];
    manifestConfig: NonNullable<BuildSourceRuntimeBundleOptions['manifestConfig']>;
    manifestQuantization: string;
    sourceQuantization: string;
    quantizationInfo?: Record<string, unknown> | null;
  } & Record<string, unknown>;
  manifestConfig: {
    hashAlgorithm: string;
  } & Record<string, unknown>;
}

export declare function assertDirectSourceRuntimeSupportedKind(
  sourceKind: unknown,
  label?: string
): DirectSourceRuntimeKind;

export declare function inferSourceQuantizationForSourceRuntime(
  tensors: SourceRuntimeTensor[] | null | undefined,
  sourceKind: string,
  options?: { logCategory?: string | null }
): string;

export declare function resolveDirectSourceRuntimePlan(options: {
  parsedArtifact: ParsedSourceArtifact;
  sourceQuantization?: string | null;
  modelKind?: string | null;
  logCategory?: string | null;
}): {
  modelType: string;
  manifestConfig: NonNullable<BuildSourceRuntimeBundleOptions['manifestConfig']>;
  manifestInference: BuildSourceRuntimeBundleOptions['inference'];
  sourceQuantization: string;
  quantizationInfo: Record<string, unknown>;
  manifestQuantization: string;
  executionVersion: 'v1';
};

export declare function resolveSourceRuntimeBundleFromParsedArtifact(
  options: ResolveSourceRuntimeBundleFromParsedArtifactOptions
): Promise<ResolvedSourceRuntimeArtifactBundle>;
