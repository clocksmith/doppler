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
  sourceKind: string | null;
  hashAlgorithm: string;
  pathSemantics: 'runtime-local' | 'artifact-relative';
  sourceFiles: SourceRuntimeShardSource[];
  auxiliaryFiles: SourceRuntimeFile[];
  tokenizer: SourceRuntimeTokenizerMetadata;
}

export const DIRECT_SOURCE_RUNTIME_MODE: 'direct-source';
export const DIRECT_SOURCE_RUNTIME_SCHEMA_VERSION: 1;
export const DIRECT_SOURCE_RUNTIME_SCHEMA: 'direct-source/v1';
export const DIRECT_SOURCE_PATH_RUNTIME_LOCAL: 'runtime-local';
export const DIRECT_SOURCE_PATH_ARTIFACT_RELATIVE: 'artifact-relative';

export function toPathKey(value: unknown): string;
export function normalizeHashAlgorithm(value: unknown): 'blake3' | 'sha256';
export function normalizeHashString(value: unknown, label: string): string | null;
export function normalizePositiveInteger(value: unknown, label: string): number;
export function normalizeAuxiliaryFiles(
  auxiliaryFiles: SourceRuntimeFile[] | null | undefined,
  defaultHashAlgorithm: string
): SourceRuntimeFile[];
export function getSourceRuntimeMetadata(
  manifest: { metadata?: Record<string, unknown> } | null | undefined
): SourceRuntimeMetadata | null;
