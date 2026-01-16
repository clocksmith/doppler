/**
 * Multi-model loader for base weights + LoRA adapters.
 *
 * @module loader/multi-model-loader
 */

import type { WeightLoadResult } from '../inference/pipelines/text/init.js';
import type { Manifest } from '../inference/pipelines/text/config.js';
import type { InferencePipeline, PipelineContexts } from '../inference/pipelines/text.js';
import type { LoRAManifest } from '../adapters/lora-loader.js';
import type { LoRAAdapter } from '../inference/pipelines/text/lora.js';
import type { RDRRManifest } from '../storage/rdrr-format.js';

export type AdapterSource =
  | LoRAAdapter
  | LoRAManifest
  | RDRRManifest
  | string;

export declare class MultiModelLoader {
  baseManifest: Manifest | null;
  baseWeights: WeightLoadResult | null;
  adapters: Map<string, LoRAAdapter>;

  loadBase(
    manifest: Manifest,
    options?: { storageContext?: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } }
  ): Promise<WeightLoadResult>;

  loadAdapter(name: string, source: AdapterSource): Promise<LoRAAdapter>;

  getAdapter(name: string): LoRAAdapter | null;

  listAdapters(): string[];

  createSharedPipeline(contexts?: PipelineContexts): Promise<InferencePipeline>;
}
