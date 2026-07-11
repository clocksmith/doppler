import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { SourceStorageContext } from '../../tooling/source-runtime-bundle.js';

export interface DopplerLoadProgress {
  phase: 'resolve' | 'manifest' | 'load' | 'ready';
  percent: number;
  message: string;
}

export interface DopplerModelSourceResolution {
  modelId: string;
  baseUrl: string | null;
  manifest: RDRRManifest | null;
  manifestHash?: string | null;
  manifestText?: string;
  storageManifest?: RDRRManifest | null;
  storageManifestText?: string;
  storageBaseUrl?: string | null;
  variantBaseUrl?: string | null;
  storageContext?: SourceStorageContext | null;
  storage?: SourceStorageContext | null;
  trace: Array<{ source: string; id: string; outcome: string }>;
}

export type DopplerModelSource =
  | string
  | {
    url: string;
    storageContext?: SourceStorageContext;
    storage?: SourceStorageContext;
    storageManifest?: RDRRManifest;
    storageBaseUrl?: string;
  }
  | {
    manifest: RDRRManifest;
    manifestText?: string;
    manifestHash?: string;
    baseUrl?: string;
    storageContext?: SourceStorageContext;
    storage?: SourceStorageContext;
    storageManifest?: RDRRManifest;
    storageBaseUrl?: string;
  };

export interface DopplerLoadOptions {
  onProgress?: (event: DopplerLoadProgress) => void;
  runtimeConfig?: Record<string, unknown>;
  isolatedLoader?: boolean;
}

export declare function createDefaultNodeLoadProgressLogger(): (event: DopplerLoadProgress) => void;

export declare function resolveLoadProgressHandlers(
  options?: DopplerLoadOptions,
  defaultLoadProgressLogger?: ((event: DopplerLoadProgress) => void) | null
): {
  userProgress: ((event: DopplerLoadProgress) => void) | null;
  pipelineProgress: ((event: DopplerLoadProgress) => void) | null;
};

export declare function fetchManifestPayloadFromBaseUrl(
  baseUrl: string
): Promise<{ text: string; manifest: RDRRManifest; manifestHash: string }>;

export declare function sha256ManifestText(value: string): Promise<string>;

export declare function resolveManifestArtifactSource(
  resolved: DopplerModelSourceResolution,
  manifestPayload: { text: string; manifest: RDRRManifest; manifestHash?: string }
): Promise<DopplerModelSourceResolution>;

export declare function resolveModelSource(model: DopplerModelSource): Promise<DopplerModelSourceResolution>;
