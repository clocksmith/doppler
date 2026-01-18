

import { loadWeights } from '../inference/pipeline/init.js';
import { parseModelConfig } from '../inference/pipeline/config.js';
import { InferencePipeline } from '../inference/pipeline.js';
import { getDopplerLoader } from './doppler-loader.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { loadLoRAFromManifest, loadLoRAFromUrl } from '../adapters/lora-loader.js';

export class MultiModelLoader {
  
  baseManifest = null;

  
  baseWeights = null;

  
  adapters = new Map();

  
  async loadBase(manifest, options = {}) {
    // Get runtime model overrides to merge with manifest inference config
    const runtimeConfig = getRuntimeConfig();
    const modelOverrides =  (runtimeConfig.inference.modelOverrides);
    const config = parseModelConfig(manifest, modelOverrides);
    this.baseManifest = manifest;
    this.baseWeights = await loadWeights(manifest, config, {
      storageContext: options.storageContext,
    });
    return this.baseWeights;
  }

  
  async loadAdapter(name, source) {
    
    let adapter;

    if (typeof source === 'string') {
      adapter = await loadLoRAFromUrl(source);
    } else if (this.#isRDRRManifest(source)) {
      const loader = getDopplerLoader();
      await loader.init();
      adapter = await loader.loadLoRAWeights(source);
    } else if (this.#isLoRAManifest(source)) {
      adapter = await loadLoRAFromManifest(source);
    } else {
      adapter = source;
    }

    const adapterName = name || adapter.name;
    this.adapters.set(adapterName, adapter);
    return adapter;
  }

  
  getAdapter(name) {
    return this.adapters.get(name) || null;
  }

  
  listAdapters() {
    return Array.from(this.adapters.keys());
  }

  
  async createSharedPipeline(contexts = {}) {
    if (!this.baseManifest || !this.baseWeights) {
      throw new Error('Base model not loaded');
    }
    const pipeline = new InferencePipeline();
    await pipeline.initialize(contexts);
    pipeline.setPreloadedWeights(this.baseWeights);
    await pipeline.loadModel(this.baseManifest);
    return pipeline;
  }

  
  #isLoRAManifest(source) {
    return typeof source === 'object' && source !== null && 'tensors' in source && 'rank' in source;
  }

  
  #isRDRRManifest(source) {
    return typeof source === 'object' && source !== null && 'shards' in source && 'modelId' in source;
  }
}
