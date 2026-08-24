import type { RuntimeModelContract } from '../../inference/runtime-model.js';
import type {
  SourceRuntimeFile,
  SourceRuntimeMetadata,
} from '../source-runtime-bundle.js';

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
  manifest: RuntimeModelContract | Record<string, unknown> | null | undefined
): SourceRuntimeMetadata | null;
