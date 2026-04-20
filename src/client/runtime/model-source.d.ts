import type { RDRRManifest } from '../../formats/rdrr/index.js';

export interface DopplerLoadProgress {
  phase: 'resolve' | 'manifest' | 'load' | 'ready';
  percent: number;
  message: string;
}

export interface DopplerModelSourceResolution {
  modelId: string;
  baseUrl: string | null;
  manifest: RDRRManifest | null;
  manifestText?: string;
  storageManifest?: RDRRManifest | null;
  storageManifestText?: string;
  storageBaseUrl?: string | null;
  variantBaseUrl?: string | null;
  trace: Array<{ source: string; id: string; outcome: string }>;
}

export type DopplerModelSource =
  | string
  | {
    url: string;
  }
  | {
    manifest: RDRRManifest;
    baseUrl?: string;
  };

export interface DopplerLoadOptions {
  onProgress?: (event: DopplerLoadProgress) => void;
  runtimeConfig?: Record<string, unknown>;
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
): Promise<{ text: string; manifest: RDRRManifest }>;

export declare function resolveManifestArtifactSource(
  resolved: DopplerModelSourceResolution,
  manifestPayload: { text: string; manifest: RDRRManifest }
): Promise<DopplerModelSourceResolution>;

export declare function resolveModelSource(model: DopplerModelSource): Promise<DopplerModelSourceResolution>;
