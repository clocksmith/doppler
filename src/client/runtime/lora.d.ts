import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { InferencePipeline } from '../../inference/pipelines/text.js';
import type { LoRAManifest } from './types.js';

export interface LoRALoadOptions {
  basePath?: string;
  resolvePath?: (path: string) => string;
  readFile?: (path: string) => Promise<ArrayBuffer | Uint8Array>;
  readOPFS?: (path: string) => Promise<ArrayBuffer | Uint8Array>;
  writeOPFS?: (path: string, data: ArrayBuffer | Uint8Array) => Promise<void>;
  fetchUrl?: (url: string) => Promise<ArrayBuffer | Uint8Array>;
}

export declare function loadLoRAAdapterForPipeline(
  pipeline: InferencePipeline | null | undefined,
  adapter: LoRAManifest | RDRRManifest | string,
  loadOptions?: LoRALoadOptions
): Promise<void>;

export declare function activateLoRAFromTrainingOutputForPipeline(
  pipeline: InferencePipeline | null | undefined,
  trainingOutput:
    | string
    | {
      adapter?: LoRAManifest | RDRRManifest | string;
      adapterManifest?: LoRAManifest | RDRRManifest;
      adapterManifestJson?: string;
      adapterManifestUrl?: string;
      adapterManifestPath?: string;
    }
    | null
    | undefined
): Promise<{
  activated: boolean;
  adapterName: string | null;
  source: string | null;
  reason: string | null;
}>;

export declare function unloadLoRAAdapterForPipeline(
  pipeline: InferencePipeline | null | undefined
): Promise<void>;

export declare function getActiveLoRAForPipeline(
  pipeline: InferencePipeline | null | undefined
): string | null;
