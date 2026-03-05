/**
 * Shared Tooling Surface Exports
 *
 * Browser-safe tooling exports shared by browser and node-facing surfaces.
 *
 * @module tooling-exports.shared
 */

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

// Shared command contract (browser + CLI parity)
export {
  TOOLING_COMMANDS,
  TOOLING_SURFACES,
  TOOLING_SUITES,
  TOOLING_VERIFY_SUITES,
  TOOLING_TRAINING_COMMAND_SCHEMA_VERSION,
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
  ensureCommandSupportedOnSurface,
} from './tooling/command-api.js';
export { runBrowserCommand, normalizeBrowserCommand } from './tooling/browser-command-runner.js';
export {
  P2P_WEBRTC_DATA_PLANE_CONTRACT_VERSION,
  isBrowserWebRTCAvailable,
  createBrowserWebRTCDataPlaneTransport,
} from './distribution/p2p-webrtc-browser.js';
export {
  P2P_CONTROL_PLANE_CONTRACT_VERSION,
  normalizeP2PControlPlaneConfig,
  resolveP2PSessionToken,
  evaluateP2PPolicyDecision,
} from './distribution/p2p-control-plane.js';
export {
  P2P_OBSERVABILITY_SCHEMA_VERSION,
  createP2PDeliveryObservabilityRecord,
  aggregateP2PDeliveryObservability,
  buildP2PAlertsFromSummary,
  buildP2PDashboardSnapshot,
} from './distribution/p2p-observability.js';

export type {
  ToolingCommand,
  ToolingSurface,
  ToolingSuite,
  ToolingIntent,
  ToolingCommandRequestInput,
  ToolingCommandRequest,
} from './tooling/command-api.js';
export type {
  BrowserCommandRunOptions,
  BrowserCommandRunResult,
} from './tooling/browser-command-runner.js';
