// ============================================================================
// Shared Tooling Surface Exports
//
// Browser-safe tooling exports shared by browser and node-facing surfaces.
// ============================================================================

// Debug
export { log } from './debug/index.js';

// Config
export {
  createConverterConfig,
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
  deleteModel,
  listModels,
} from './storage/shard-manager.js';
export { exportModelToDirectory } from './storage/export.js';
export { ensureModelCached } from './tooling/opfs-cache.js';
export { parseManifest, getManifest, setManifest, clearManifest, classifyTensorRole } from './formats/rdrr/index.js';
export { inferEmbeddingOutputConfig } from './converter/core.js';

// GPU init + capabilities
export { initDevice, getDevice, getKernelCapabilities, getPlatformConfig, isWebGPUAvailable } from './gpu/device.js';

// Consumer-preseeded shader sources (for bundle-friendly builds that avoid
// runtime HTTP fetch of WGSL kernels).
export { registerShaderSources, hasPreseededShaderSource } from './gpu/kernels/shader-cache.js';

// Memory tooling
export { captureMemorySnapshot } from './loader/memory-monitor.js';
export { destroyBufferPool } from './memory/buffer-pool.js';

// Browser-safe runtime profile helpers
export {
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimeProfile,
  applyRuntimeProfile,
} from './inference/browser-harness-runtime-helpers.js';

// Shared command contract (browser + CLI parity)
export {
  TOOLING_COMMANDS,
  TOOLING_SURFACES,
  TOOLING_WORKLOADS,
  TOOLING_VERIFY_WORKLOADS,
  TOOLING_TRAINING_COMMAND_SCHEMA_VERSION,
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './tooling/command-api.js';
export { runBrowserCommand, normalizeBrowserCommand } from './tooling/browser-command-runner.js';
