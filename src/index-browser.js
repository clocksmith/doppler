export { DOPPLER_VERSION } from './version.js';
export { doppler } from './client/doppler-api.browser.js';

// Core loaders
export {
  DopplerLoader,
  getDopplerLoader,
  createDopplerLoader,
} from './loader/doppler-loader.js';
export { MultiModelLoader } from './loader/multi-model-loader.js';

// Inference pipeline
export { InferencePipeline, EmbeddingPipeline, createPipeline } from './generation/index.js';
export {
  StructuredJsonHeadPipeline,
  isStructuredJsonHeadModelType,
  createStructuredJsonHeadPipeline,
  DreamStructuredPipeline,
  isDreamStructuredModelType,
  createDreamStructuredPipeline,
} from './generation/index.js';
export {
  EnergyRowHeadPipeline,
  createEnergyRowHeadPipeline,
  DreamEnergyHeadPipeline,
  createDreamEnergyHeadPipeline,
} from './inference/pipelines/energy-head/row-head-pipeline.js';
export { KVCache } from './inference/kv-cache.js';
export { Tokenizer } from './inference/tokenizer.js';
export { SpeculativeDecoder } from './inference/speculative.js';

// Multi-model orchestration
export { ExpertRouter } from './inference/expert-router.js';
export { MoERouter } from './inference/moe-router.js';
export { MultiModelNetwork } from './inference/multi-model-network.js';
export { MultiPipelinePool } from './inference/multi-pipeline-pool.js';

// GPU primitives
export {
  LogitMergeKernel,
  getLogitMergeKernel,
  mergeLogits,
  mergeMultipleLogits,
} from './gpu/kernels/logit-merge.js';

// LoRA Adapter Infrastructure
export {
  ADAPTER_MANIFEST_SCHEMA,
  validateManifest as validateAdapterManifest,
  parseManifest as parseAdapterManifest,
  serializeManifest as serializeAdapterManifest,
  createManifest as createAdapterManifest,
  computeLoRAScale,
  loadLoRAWeights,
  loadLoRAFromManifest,
  loadLoRAFromUrl,
  loadLoRAFromSafetensors,
  AdapterManager,
  getAdapterManager,
  resetAdapterManager,
  AdapterRegistry,
  getAdapterRegistry,
  resetAdapterRegistry,
  createMemoryRegistry,
} from './adapters/index.js';

// Tooling Surface — re-exported for browser tooling consumers.
export * from './tooling-exports.browser.js';
