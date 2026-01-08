/**
 * Multi-model loader for base weights + LoRA adapters.
 *
 * @module loader/multi-model-loader
 */

import { loadWeights } from '../inference/pipeline/init.js';
import { parseModelConfig } from '../inference/pipeline/config.js';
import { InferencePipeline } from '../inference/pipeline.js';
import { getDopplerLoader } from './doppler-loader.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { loadLoRAFromManifest, loadLoRAFromUrl } from '../adapters/lora-loader.js';

export class MultiModelLoader {
  /** @type {import('../inference/pipeline/config.js').Manifest | null} */
  baseManifest = null;

  /** @type {import('../inference/pipeline/init.js').WeightLoadResult | null} */
  baseWeights = null;

  /** @type {Map<string, import('../inference/pipeline/lora.js').LoRAAdapter>} */
  adapters = new Map();

  /**
   * @param {import('../inference/pipeline/config.js').Manifest} manifest
   * @param {{ storageContext?: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } }} [options={}]
   * @returns {Promise<import('../inference/pipeline/init.js').WeightLoadResult>}
   */
  async loadBase(manifest, options = {}) {
    // Get runtime model overrides to merge with manifest inference config
    const runtimeConfig = getRuntimeConfig();
    const modelOverrides = /** @type {import('../config/schema/index.js').ModelInferenceOverrides | undefined} */ (runtimeConfig.inference.modelOverrides);
    const config = parseModelConfig(manifest, modelOverrides);
    this.baseManifest = manifest;
    this.baseWeights = await loadWeights(manifest, config, {
      storageContext: options.storageContext,
      verifyHashes: false,
    });
    return this.baseWeights;
  }

  /**
   * @param {string} name
   * @param {import('./multi-model-loader.js').AdapterSource} source
   * @returns {Promise<import('../inference/pipeline/lora.js').LoRAAdapter>}
   */
  async loadAdapter(name, source) {
    /** @type {import('../inference/pipeline/lora.js').LoRAAdapter} */
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

  /**
   * @param {string} name
   * @returns {import('../inference/pipeline/lora.js').LoRAAdapter | null}
   */
  getAdapter(name) {
    return this.adapters.get(name) || null;
  }

  /**
   * @returns {string[]}
   */
  listAdapters() {
    return Array.from(this.adapters.keys());
  }

  /**
   * @param {import('../inference/pipeline.js').PipelineContexts} [contexts={}]
   * @returns {Promise<InferencePipeline>}
   */
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

  /**
   * @param {import('./multi-model-loader.js').AdapterSource} source
   * @returns {source is import('../adapters/lora-loader.js').LoRAManifest}
   */
  #isLoRAManifest(source) {
    return typeof source === 'object' && source !== null && 'tensors' in source && 'rank' in source;
  }

  /**
   * @param {import('./multi-model-loader.js').AdapterSource} source
   * @returns {source is import('../storage/rdrr-format.js').RDRRManifest}
   */
  #isRDRRManifest(source) {
    return typeof source === 'object' && source !== null && 'shards' in source && 'modelId' in source;
  }
}
