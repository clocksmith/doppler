
// ============================================================================
// Tooling Surface Exports
//
// Internal tooling, diagnostics, and infrastructure used by demo, CLI,
// and harness code. Not part of the core inference API.
//
// Consumers can import from 'doppler/tooling' or individual deep paths.
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
export { parseManifest, getManifest, setManifest, clearManifest, classifyTensorRole } from './formats/rdrr/index.js';

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
