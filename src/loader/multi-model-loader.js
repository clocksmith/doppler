

import { loadWeights } from '../inference/pipelines/text/init.js';
import { parseModelConfig } from '../inference/pipelines/text/config.js';
import { InferencePipeline } from '../inference/pipelines/text.js';
import { getDopplerLoader } from './doppler-loader.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { loadLoRAFromManifest, loadLoRAFromUrl } from '../experimental/adapters/lora-loader.js';

export class MultiModelLoader {
  
  baseManifest = null;

  
  baseWeights = null;

  
  adapters = new Map();

  #pipelines = new Set();

  async _loadBaseWeights(manifest, options, runtimeConfig) {
    const modelOverrides =  (runtimeConfig.inference.modelOverrides);
    const config = parseModelConfig(manifest, modelOverrides);
    return loadWeights(manifest, config, {
      storageContext: options.storageContext,
      keepF32Weights: runtimeConfig.inference.compute.keepF32Weights === true,
    });
  }

  async _resolveAdapterSource(source) {
    if (typeof source === 'string') {
      return loadLoRAFromUrl(source);
    }
    if (this.#isRDRRManifest(source)) {
      const loader = getDopplerLoader();
      await loader.init();
      return loader.loadLoRAWeights(source);
    }
    if (this.#isLoRAManifest(source)) {
      return loadLoRAFromManifest(source);
    }
    return source;
  }

  _createPipeline() {
    return new InferencePipeline();
  }

  _getBaseLoader() {
    return getDopplerLoader();
  }

  async unload() {
    const pipelines = Array.from(this.#pipelines);
    this.#pipelines.clear();
    await Promise.all(pipelines.map(async (pipeline) => pipeline.unload()));

    if (this.baseWeights) {
      const loader = this._getBaseLoader();
      await loader.unload();
    }

    this.baseManifest = null;
    this.baseWeights = null;
    this.adapters.clear();
  }

  async loadBase(manifest, options = {}) {
    await this.unload();

    const runtimeConfig = getRuntimeConfig();
    const weights = await this._loadBaseWeights(manifest, options, runtimeConfig);
    this.baseManifest = manifest;
    this.baseWeights = weights;
    return weights;
  }

  async loadAdapter(name, source) {
    const adapter = await this._resolveAdapterSource(source);

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
    const pipeline = this._createPipeline();
    const unloadPipeline = pipeline.unload.bind(pipeline);
    pipeline.unload = async () => {
      try {
        await unloadPipeline();
      } finally {
        this.#pipelines.delete(pipeline);
      }
    };

    try {
      await pipeline.initialize(contexts);
      pipeline.setPreloadedWeights(this.baseWeights);
      await pipeline.loadModel(this.baseManifest);
      this.#pipelines.add(pipeline);
      return pipeline;
    } catch (error) {
      await pipeline.unload().catch(() => {});
      throw error;
    }
  }

  
  #isLoRAManifest(source) {
    return typeof source === 'object' && source !== null && 'tensors' in source && 'rank' in source;
  }

  
  #isRDRRManifest(source) {
    return typeof source === 'object' && source !== null && 'shards' in source && 'modelId' in source;
  }
}
