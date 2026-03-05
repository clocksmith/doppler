export declare const DOPPLER_VERSION: string;

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

// Types
export type { RDRRManifest, ShardInfo } from './formats/rdrr/index.js';
export type { TensorLocation, LoadProgress, LoadOptions, LoaderStats } from './loader/doppler-loader.js';
export type { AdapterSource } from './loader/multi-model-loader.js';
export type { ParsedModelConfig } from './generation/index.js';
export type { SamplingOptions } from './generation/index.js';
export type {
  GenerateOptions,
  GenerationResult,
  KVCacheSnapshot,
  LayerWeights,
  ExpertWeights,
  RouterWeights,
} from './generation/index.js';
export type { LoRAAdapter, LoRAModuleName } from './generation/index.js';
export type { ExpertNode, ExpertTask } from './inference/multi-model-network.js';

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

export type {
  AdapterManifest,
  AdapterMetadata,
  AdapterTensorSpec,
  LoRALoadOptions,
  LoRAWeightsResult,
  AdapterState,
  EnableAdapterOptions,
  AdapterStackOptions,
  AdapterManagerEvents,
  AdapterRegistryEntry,
  AdapterQueryOptions,
} from './adapters/index.js';

// ============================================================================
// Tooling Surface — Re-exported for backward compatibility.
// Prefer importing from 'doppler/tooling' for tooling-only consumers.
// ============================================================================

export * from './tooling-exports.shared.js';
