import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { ManifestEmbeddingPostprocessorSchema } from '../config/schema/index.js';
import type {
  BuildSourceRuntimeBundleOptions,
  BuildSourceRuntimeBundleResult,
  SourceRuntimeFile,
  SourceRuntimeTensor,
} from './source-runtime-bundle.js';

export declare const SOURCE_ARTIFACT_KIND_SAFETENSORS: 'safetensors';
export declare const SOURCE_ARTIFACT_KIND_GGUF: 'gguf';
export declare const SOURCE_ARTIFACT_KIND_TFLITE: 'tflite';

export type SourceArtifactKind =
  | typeof SOURCE_ARTIFACT_KIND_SAFETENSORS
  | typeof SOURCE_ARTIFACT_KIND_GGUF
  | typeof SOURCE_ARTIFACT_KIND_TFLITE;

export type DirectSourceRuntimeKind =
  | typeof SOURCE_ARTIFACT_KIND_SAFETENSORS
  | typeof SOURCE_ARTIFACT_KIND_GGUF;

export interface ParsedSourceArtifact {
  sourceKind: SourceArtifactKind | string;
  config: Record<string, unknown>;
  tensors: SourceRuntimeTensor[];
  architectureHint?: string | null;
  embeddingPostprocessor?: ManifestEmbeddingPostprocessorSchema | null;
  architecture: Record<string, unknown> | string | null;
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
  quantization?: Record<string, unknown> | null;
  hashFileEntries: (
    entries: SourceRuntimeFile[] | null | undefined,
    hashAlgorithm: string
  ) => Promise<SourceRuntimeFile[]>;
}

export interface ResolvedSourceRuntimeArtifactBundle extends BuildSourceRuntimeBundleResult {
  manifest: RDRRManifest;
  sourceKind: DirectSourceRuntimeKind;
  sourceQuantization: string;
  sourceFiles: SourceRuntimeFile[];
  auxiliaryFiles: SourceRuntimeFile[];
  hashAlgorithm: string;
  modelId: string;
  plan: {
    modelType: string;
    manifestInference: BuildSourceRuntimeBundleOptions['inference'];
    quantizationInfo?: Record<string, unknown> | null;
  } & Record<string, unknown>;
  converterConfig: {
    manifest: {
      hashAlgorithm: string;
    } & Record<string, unknown>;
  } & Record<string, unknown>;
}

export declare function normalizeSourceArtifactKind(value: unknown): string | null;

export declare function assertDirectSourceRuntimeSupportedKind(
  sourceKind: unknown,
  label?: string
): DirectSourceRuntimeKind;

export declare function assertSupportedSourceDtypes(
  tensors: SourceRuntimeTensor[] | null | undefined,
  sourceKind: string
): void;

export declare function inferSourceQuantizationForSourceRuntime(
  tensors: SourceRuntimeTensor[] | null | undefined,
  sourceKind: string,
  options?: { logCategory?: string | null }
): string;

export declare function resolveSourceRuntimeComputePrecision(
  tensors: SourceRuntimeTensor[] | null | undefined,
  sourceQuantization: string | null | undefined
): 'f16' | 'f32';

export declare function resolveSourceRuntimeModelIdHint(options: {
  requestedModelId?: string | null;
  plan: { quantizationInfo?: Record<string, unknown> | null };
  sourceKind: string;
  sourcePath?: string | null;
  label?: string | null;
}): string;

export declare function resolveSourceRuntimeBundleFromParsedArtifact(
  options: ResolveSourceRuntimeBundleFromParsedArtifactOptions
): Promise<ResolvedSourceRuntimeArtifactBundle>;
