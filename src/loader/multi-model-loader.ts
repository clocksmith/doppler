/**
 * Multi-model loader for base weights + LoRA adapters.
 *
 * @module loader/multi-model-loader
 */

import { loadWeights, type WeightLoadResult } from '../inference/pipeline/init.js';
import { parseModelConfig, type Manifest } from '../inference/pipeline/config.js';
import { InferencePipeline, type PipelineContexts } from '../inference/pipeline.js';
import { getDopplerLoader } from './doppler-loader.js';
import { getRuntimeConfig } from '../config/runtime.js';
import type { ModelInferenceOverrides } from '../config/schema/index.js';
import { loadLoRAFromManifest, loadLoRAFromUrl, type LoRAManifest } from '../adapters/lora-loader.js';
import type { LoRAAdapter } from '../inference/pipeline/lora.js';
import type { RDRRManifest } from '../storage/rdrr-format.js';

export type AdapterSource =
  | LoRAAdapter
  | LoRAManifest
  | RDRRManifest
  | string;

export class MultiModelLoader {
  baseManifest: Manifest | null = null;
  baseWeights: WeightLoadResult | null = null;
  adapters = new Map<string, LoRAAdapter>();

  async loadBase(
    manifest: Manifest,
    options: { storageContext?: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } } = {}
  ): Promise<WeightLoadResult> {
    // Get runtime model overrides to merge with manifest inference config
    const runtimeConfig = getRuntimeConfig();
    const modelOverrides = runtimeConfig.inference.modelOverrides as ModelInferenceOverrides | undefined;
    const config = parseModelConfig(manifest, modelOverrides);
    this.baseManifest = manifest;
    this.baseWeights = await loadWeights(manifest, config, {
      storageContext: options.storageContext,
      verifyHashes: false,
    });
    return this.baseWeights;
  }

  async loadAdapter(name: string, source: AdapterSource): Promise<LoRAAdapter> {
    let adapter: LoRAAdapter;

    if (typeof source === 'string') {
      adapter = await loadLoRAFromUrl(source);
    } else if (this.isRDRRManifest(source)) {
      const loader = getDopplerLoader();
      await loader.init();
      adapter = await loader.loadLoRAWeights(source);
    } else if (this.isLoRAManifest(source)) {
      adapter = await loadLoRAFromManifest(source);
    } else {
      adapter = source;
    }

    const adapterName = name || adapter.name;
    this.adapters.set(adapterName, adapter);
    return adapter;
  }

  getAdapter(name: string): LoRAAdapter | null {
    return this.adapters.get(name) || null;
  }

  listAdapters(): string[] {
    return Array.from(this.adapters.keys());
  }

  async createSharedPipeline(
    contexts: PipelineContexts = {}
  ): Promise<InferencePipeline> {
    if (!this.baseManifest || !this.baseWeights) {
      throw new Error('Base model not loaded');
    }
    const pipeline = new InferencePipeline();
    await pipeline.initialize(contexts);
    pipeline.setPreloadedWeights(this.baseWeights);
    await pipeline.loadModel(this.baseManifest);
    return pipeline;
  }

  private isLoRAManifest(source: AdapterSource): source is LoRAManifest {
    return typeof source === 'object' && source !== null && 'tensors' in source && 'rank' in source;
  }

  private isRDRRManifest(source: AdapterSource): source is RDRRManifest {
    return typeof source === 'object' && source !== null && 'shards' in source && 'modelId' in source;
  }
}
