import type { IntegrityExtensions, RDRRManifest, TensorMap } from '../formats/rdrr/types.js';
import type { IntegrityBuildProgress } from '../formats/rdrr/integrity.js';

export interface NormalizedManifestLoweringEntry {
  kernelRef: string;
  backend: string;
  targetDescriptorCorrectnessHash: string | null;
  frontendVersion: string | null;
  tsirSemanticDigest: string | null;
  tsirRealizationDigest: string | null;
  emitterDigest: string | null;
  compilerVersion: string | null;
  exactness: Record<string, unknown> | null;
  rejectionReasons: string[];
}

export declare function normalizeManifestLoweringEntry(
  entry: unknown,
  label?: string
): NormalizedManifestLoweringEntry;

export declare function buildManifestIntegrityFromModelDir(
  manifest: RDRRManifest,
  options: {
    modelDir: string;
    tensorMap?: TensorMap;
    blockSize?: number;
    onProgress?: (progress: IntegrityBuildProgress) => void;
    readRange?: (filePath: string, offset: number, length: number) => Promise<Uint8Array | ArrayBuffer>;
    hashBlockBytesSha256?: (bytes: Uint8Array) => string;
  }
): Promise<{
  integrityExtensions: IntegrityExtensions;
  integrityExtensionsHash: string;
}>;

export declare function refreshManifestIntegrity(options: {
  modelDir: string;
  manifestPath?: string | null;
  blockSize?: number;
  dryRun?: boolean;
  skipShardCheck?: boolean;
  onProgress?: (progress: IntegrityBuildProgress) => void;
}): Promise<{
  manifestPath: string;
  manifest: RDRRManifest;
  integrityExtensions: IntegrityExtensions;
  integrityExtensionsHash: string;
  wrote: boolean;
}>;
