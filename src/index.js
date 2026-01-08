export const DOPPLER_VERSION = '0.1.0';

// Core loaders
export {
  DopplerLoader,
  getDopplerLoader,
  createDopplerLoader,
} from './loader/doppler-loader.js';
export { MultiModelLoader } from './loader/multi-model-loader.js';

// Inference pipeline
export { InferencePipeline, createPipeline } from './inference/pipeline.js';
export { KVCache } from './inference/kv-cache.js';
export { Tokenizer } from './inference/tokenizer.js';
export { SpeculativeDecoder } from './inference/speculative.js';

// Multi-model orchestration
export { ExpertRouter } from './inference/expert-router.js';
export { MoERouter } from './inference/moe-router.js';
export { MultiModelNetwork } from './inference/multi-model-network.js';
export { MultiPipelinePool } from './inference/multi-pipeline-pool.js';
export { evolveNetwork, mutateGenome, crossoverGenome } from './inference/network-evolution.js';

// LoRA Adapter Infrastructure (Tier 1 P0 - RSI Foundation)
export {
  // Manifest
  ADAPTER_MANIFEST_SCHEMA,
  validateManifest,
  parseManifest,
  serializeManifest,
  createManifest,
  computeLoRAScale,
  // Loader
  loadLoRAWeights,
  loadLoRAFromManifest,
  loadLoRAFromUrl,
  loadLoRAFromSafetensors,
  // Manager
  AdapterManager,
  getAdapterManager,
  resetAdapterManager,
  // Registry
  AdapterRegistry,
  getAdapterRegistry,
  resetAdapterRegistry,
  createMemoryRegistry,
} from './adapters/index.js';
