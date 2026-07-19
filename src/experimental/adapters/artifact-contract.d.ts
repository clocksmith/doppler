import type { AdapterManifest } from './adapter-manifest.js';

export type AdapterArtifactLifecycle = 'preserved' | 'candidate' | 'qualified' | 'promoted' | 'revoked';
export type AdapterArtifactAccess = 'public' | 'gated' | 'private';

export interface HuggingFaceArtifactOrigin {
  provider: 'huggingface';
  repoId: string;
  revision: string;
  path: string;
}

export interface GcsArtifactOrigin {
  provider: 'gcs';
  bucket: string;
  object: string;
  generation: string;
}

export interface HttpsPreservationOrigin {
  provider: 'https-preservation';
  url: string;
  revision: string;
}

export type AdapterArtifactOrigin = HuggingFaceArtifactOrigin | GcsArtifactOrigin;
export type AdapterArtifactMirror = AdapterArtifactOrigin | HttpsPreservationOrigin;

export interface AdapterArtifactRecord {
  schema: 'doppler.adapter-artifact/v1';
  artifactId: string;
  lifecycle: AdapterArtifactLifecycle;
  access: AdapterArtifactAccess;
  weights: {
    sha256: string;
    bytes: number;
    format: 'safetensors';
  };
  adapterManifest: AdapterManifest;
  trainingBase: {
    repoId: string;
    revision: string;
  };
  runtimeBase: {
    modelId: string;
    modelSha256: string;
    manifestSha256: string;
    tokenizerSha256: string;
    weightPackId: string;
    weightPackSha256: string;
    manifestVariantId: string;
    conversionConfigSha256: string;
  };
  primaryOrigin: AdapterArtifactOrigin | null;
  preservationMirrors: AdapterArtifactMirror[];
  evidence: Array<{ kind: string; path: string; sha256: string }>;
  claimBoundary?: string;
}

export interface AdapterArtifactValidationError {
  field: string;
  message: string;
  value?: unknown;
}

export interface AdapterArtifactValidationResult {
  valid: boolean;
  errors: AdapterArtifactValidationError[];
}

export declare const ADAPTER_ARTIFACT_SCHEMA: 'doppler.adapter-artifact/v1';
export declare const ADAPTER_ARTIFACT_LIFECYCLES: readonly AdapterArtifactLifecycle[];
export declare const ADAPTER_ARTIFACT_ACCESS: readonly AdapterArtifactAccess[];
export declare const ADAPTER_ARTIFACT_ORIGIN_PROVIDERS: readonly ['huggingface', 'gcs', 'https-preservation'];

export declare function validateAdapterArtifactOrigin(
  origin: unknown,
  options?: { allowPreservation?: boolean }
): AdapterArtifactValidationResult;
export declare function validateAdapterArtifactRecord(record: unknown): AdapterArtifactValidationResult;
export declare function assertAdapterArtifactRecord<T extends AdapterArtifactRecord>(record: T): T;
export declare function buildImmutableArtifactUrl(origin: AdapterArtifactOrigin): string;
export declare function adapterArtifactCacheKey(record: AdapterArtifactRecord): string;
