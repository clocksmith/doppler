import type { RDRRManifest } from '../formats/rdrr/types.js';
import type { SourceRuntimeShardSource, SourceStorageContext } from '../tooling/source-runtime-bundle.js';

export declare const ARTIFACT_FORMAT_RDRR: 'rdrr';
export declare const ARTIFACT_FORMAT_DIRECT_SOURCE: 'direct-source';

export type ArtifactFormat = typeof ARTIFACT_FORMAT_RDRR | typeof ARTIFACT_FORMAT_DIRECT_SOURCE;

export interface CreateArtifactStorageContextOptions {
  manifest: RDRRManifest;
  expectedFormat?: ArtifactFormat | null;
  shardSources?: SourceRuntimeShardSource[] | null;
  readRange: (
    path: string,
    offset: number,
    length: number | null
  ) => Promise<ArrayBuffer | Uint8Array>;
  streamRange?: (
    path: string,
    offset: number,
    length: number,
    options?: { chunkBytes?: number }
  ) => AsyncIterable<ArrayBuffer | Uint8Array>;
  readText?: (path: string) => Promise<string | Record<string, unknown> | null | undefined>;
  readBinary?: (path: string) => Promise<ArrayBuffer | Uint8Array | null | undefined>;
  tokenizerJsonPath?: string | null;
  tokenizerModelPath?: string | null;
  verifyHashes?: boolean;
  hashesTrusted?: boolean;
}

export interface CreateHttpArtifactStorageContextOptions {
  verifyHashes?: boolean;
}

export declare function getArtifactFormat(
  manifest: RDRRManifest | Record<string, unknown> | null | undefined
): ArtifactFormat | null;

export declare function createArtifactStorageContext(
  options: CreateArtifactStorageContextOptions
): SourceStorageContext;

export declare function createNodeFileArtifactStorageContext(
  baseUrl: string | null | undefined,
  manifest: RDRRManifest
): SourceStorageContext | null;

export declare function createHttpArtifactStorageContext(
  baseUrl: string | null | undefined,
  manifest: RDRRManifest,
  options?: CreateHttpArtifactStorageContextOptions
): SourceStorageContext | null;
