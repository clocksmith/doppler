// ============================================================================
// Shard Manager (primary storage orchestrator)
// ============================================================================

export {
  getManifest,
  setOpfsPathConfig,
  getOpfsPathConfig,
  getHashAlgorithm,
  hexToBytes,
  computeBlake3,
  computeSHA256,
  computeHash,
  createStreamingHasher,
  getStorageCapabilities,
  getStorageBackendType,
  initStorage,
  openModelStore,
  getCurrentModelId,
  writeShard,
  createShardWriter,
  createConversionShardWriter,
  createFileWriter,
  loadShard,
  loadShardRange,
  streamShardRange,
  loadShardSync,
  shardExists,
  getShardStoredSize,
  verifyIntegrity,
  deleteShard,
  deleteModel,
  listModels,
  listFilesInStore,
  loadFileFromStore,
  loadFileRangeFromStore,
  streamFileFromStore,
  getModelInfo,
  modelExists,
  saveManifest,
  loadManifestFromStore,
  loadTensorsFromStore,
  saveTensorsToStore,
  saveTokenizer,
  loadTokenizerFromStore,
  saveTokenizerModel,
  loadTokenizerModelFromStore,
  saveAuxFile,
  loadAuxFile,
  loadAuxText,
  deleteFileFromStore,
  cleanup,
} from './shard-manager.js';

// ============================================================================
// Model Registry
// ============================================================================

export {
  loadModelRegistry,
  saveModelRegistry,
  listRegisteredModels,
  registerModel,
  removeRegisteredModel,
} from './registry.js';

// ============================================================================
// Storage Inventory
// ============================================================================

export {
  listStorageInventory,
  deleteStorageEntry,
} from './inventory.js';

// ============================================================================
// Quota & Storage Detection
// ============================================================================

export {
  isStorageAPIAvailable,
  isOPFSAvailable,
  isIndexedDBAvailable,
  getQuotaInfo,
  isPersisted,
  requestPersistence,
  checkSpaceAvailable,
  formatBytes,
  getStorageReport,
  QuotaExceededError,
  monitorStorage,
  getSuggestions,
  clearCache,
} from './quota.js';

// ============================================================================
// Downloader
// ============================================================================

export {
  downloadModel,
  pauseDownload,
  resumeDownload,
  getDownloadProgress,
  listDownloads,
  cancelDownload,
  checkDownloadNeeded,
  formatSpeed,
  estimateTimeRemaining,
  persistDownloadedShardIfNeeded,
} from './downloader.js';

// ============================================================================
// Quickstart Downloader
// ============================================================================

export {
  setCDNBaseUrl,
  getCDNBaseUrl,
  QUICKSTART_MODELS,
  getQuickStartModel,
  listQuickStartModels,
  registerQuickStartModel,
  downloadQuickStartModel,
  isModelDownloaded,
  getModelDownloadSize,
  formatModelInfo,
} from './quickstart-downloader.js';

// ============================================================================
// Export
// ============================================================================

export { exportModelToDirectory } from './export.js';

// ============================================================================
// Preflight
// ============================================================================

export {
  GEMMA_1B_REQUIREMENTS,
  MODEL_REQUIREMENTS,
  runPreflightChecks,
  formatPreflightResult,
} from './preflight.js';

// ============================================================================
// Reports
// ============================================================================

export { saveReport } from './reports.js';

// ============================================================================
// Download Types
// ============================================================================

export {
  DB_NAME,
  DB_VERSION,
  STORE_NAME,
  getDistributionConfig,
  getDefaultConcurrency,
  getMaxRetries,
  getInitialRetryDelayMs,
  getMaxRetryDelayMs,
  getCdnBasePath,
  getProgressUpdateIntervalMs,
  getRequiredContentEncoding,
} from './download-types.js';

// ============================================================================
// Blake3
// ============================================================================

export {
  createHasher,
  hash,
} from './blake3.js';

// ============================================================================
// Emulated VRAM
// ============================================================================

export {
  EmulatedVramStore,
  createEmulatedVramStore,
  detectLocalResources,
} from './emulated-vram.js';
