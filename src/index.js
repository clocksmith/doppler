export const DOPPLER_VERSION = '0.1.0';

// Core loaders
export {
  DopplerLoader,
  getDopplerLoader,
  createDopplerLoader,
} from './loader/doppler-loader.js';
export { MultiModelLoader } from './loader/multi-model-loader.js';

// Inference pipeline
export { InferencePipeline, EmbeddingPipeline, createPipeline } from './inference/pipelines/text.js';
export {
  DreamStructuredPipeline,
  isDreamStructuredModelType,
  createDreamStructuredPipeline,
} from './inference/pipelines/dream/pipeline.js';
export {
  DreamEnergyHeadPipeline,
  createDreamEnergyHeadPipeline,
} from './inference/pipelines/dream/energy-head-pipeline.js';
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

// LoRA Adapter Infrastructure (Tier 1 P0 - RSI Foundation)
export {
  // Manifest
  ADAPTER_MANIFEST_SCHEMA,
  validateManifest as validateAdapterManifest,
  parseManifest as parseAdapterManifest,
  serializeManifest as serializeAdapterManifest,
  createManifest as createAdapterManifest,
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

// ============================================================================
// Public Tooling Surface (used by the demo + diagnostics)
// ============================================================================

// Debug
export { log } from './debug/index.js';

// Config
export {
  listPresets,
  createConverterConfig,
  detectPreset,
  resolvePreset,
} from './config/index.js';
export { getRuntimeConfig, setRuntimeConfig } from './config/runtime.js';
export { DEFAULT_MANIFEST_INFERENCE } from './config/schema/index.js';
export { TOOLING_INTENTS } from './config/schema/tooling.schema.js';

// Storage + manifests
export { formatBytes, getQuotaInfo } from './storage/quota.js';
export { listRegisteredModels, registerModel, removeRegisteredModel } from './storage/registry.js';
export { listStorageInventory, deleteStorageEntry } from './storage/inventory.js';
export {
  openModelStore,
  writeShard,
  loadManifestFromStore,
  loadShard,
  loadTensorsFromStore,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  loadTokenizerFromStore,
  loadTokenizerModelFromStore,
  listFilesInStore,
  loadFileFromStore,
  streamFileFromStore,
  computeHash,
} from './storage/shard-manager.js';
export { exportModelToDirectory } from './storage/export.js';
export { parseManifest, getManifest, setManifest, clearManifest, classifyTensorRole } from './storage/rdrr-format.js';

// Browser conversion + file pickers
export { convertModel, createRemoteModelSources, isConversionSupported } from './browser/browser-converter.js';
export { pickModelDirectory, pickModelFiles } from './browser/file-picker.js';
export { buildManifestInference, inferEmbeddingOutputConfig } from './converter/manifest-inference.js';

// GPU init + capabilities
export { initDevice, getDevice, getKernelCapabilities, getPlatformConfig, isWebGPUAvailable } from './gpu/device.js';

// Memory tooling
export { captureMemorySnapshot } from './loader/memory-monitor.js';
export { destroyBufferPool } from './memory/buffer-pool.js';

// Diagnostics harness
export { loadRuntimePreset, applyRuntimePreset, runBrowserSuite } from './inference/browser-harness.js';

// Energy utilities (used by demo energy mode)
export { buildLayout, getDefaultSpec, buildVliwDatasetFromSpec } from './inference/pipelines/energy/vliw-generator.js';

// Shared command contract (browser + CLI parity)
export {
  TOOLING_COMMANDS,
  TOOLING_SURFACES,
  TOOLING_SUITES,
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
  ensureCommandSupportedOnSurface,
} from './tooling/command-api.js';
export { runBrowserCommand, normalizeBrowserCommand } from './tooling/browser-command-runner.js';
export { runNodeCommand, normalizeNodeCommand, hasNodeWebGPUSupport } from './tooling/node-command-runner.js';
export { runBrowserCommandInNode, normalizeNodeBrowserCommand } from './tooling/node-browser-command-runner.js';
