import type { RDRRManifest } from '../formats/rdrr/types.js';
import type {
  SourceRuntimeMetadata,
  SourceStorageContext,
} from '../tooling/source-runtime-bundle.js';

export interface StoredSourceArtifactFile {
  path: string;
  size: number | null;
  hash: string | null;
  hashAlgorithm: string | null;
  kind: string;
}

export interface StoredSourceArtifactSourceFile extends StoredSourceArtifactFile {
  index: number;
}

export interface StoredSourceArtifactDescriptor {
  sourceRuntime: SourceRuntimeMetadata;
  sourceFiles: StoredSourceArtifactSourceFile[];
  auxiliaryFiles: StoredSourceArtifactFile[];
  files: StoredSourceArtifactFile[];
  totalBytes: number;
  fingerprint: string;
}

export interface StoredSourceArtifactIntegrity {
  valid: boolean;
  missingFiles: string[];
  corruptFiles: string[];
}

export interface StoredSourceArtifactManifestSynthesisResult {
  manifest: RDRRManifest;
  changed: boolean;
}

export declare function normalizeSourceArtifactPath(value: unknown): string;

export declare function listSourceArtifactFiles(manifest: RDRRManifest): StoredSourceArtifactFile[];

export declare function resolveSourceArtifact(
  manifest: RDRRManifest
): StoredSourceArtifactDescriptor | null;

export declare function buildSourceArtifactFingerprint(manifest: RDRRManifest): string | null;

export declare function synthesizeStoredSourceArtifactManifest(
  manifest: RDRRManifest
): StoredSourceArtifactManifestSynthesisResult;

export declare function verifyStoredSourceArtifact(
  manifest: RDRRManifest,
  options?: { checkHashes?: boolean }
): Promise<StoredSourceArtifactIntegrity>;

export declare function createStoredSourceArtifactContext(
  manifest: RDRRManifest,
  options?: { verifyHashes?: boolean }
): SourceStorageContext;
