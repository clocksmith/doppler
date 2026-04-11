import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { InferencePipeline } from '../../inference/pipelines/text.js';

export declare function loadLoRAAdapterForPipeline(
  pipeline: InferencePipeline | null | undefined,
  adapter: LoRAManifest | RDRRManifest | string
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
