// ============================================================================
// Core Loader
// ============================================================================

export {
  DopplerLoader,
  getDopplerLoader,
  createDopplerLoader,
} from './doppler-loader.js';

export { MultiModelLoader } from './multi-model-loader.js';

export { LoaderState, createLoaderState } from './loader-state.js';

// ============================================================================
// Manifest & Shard Resolution
// ============================================================================

export {
  needsNormWeightOffset,
  getLargeWeightConfig,
  getLargeWeightMaxBytes,
  estimateMatmulWeightBytes,
  resolveWeightLayout,
  shouldStreamLargeWeight,
  isMoEModel,
} from './manifest-config.js';

export { buildTensorLocations } from './shard-resolver.js';

export { ShardCache, createShardCache } from './shard-cache.js';

// ============================================================================
// Expert Loading & Caching
// ============================================================================

export {
  preloadShardsForExpert,
  prefetchExperts,
  predictNextLayerExperts,
  loadExpert,
} from './experts/expert-loader.js';

export {
  ExpertCache,
  getExpertCache,
  createExpertCache,
} from './experts/expert-cache.js';

// ============================================================================
// Tensor Loading
// ============================================================================

export {
  isPackedQ4K,
  shouldUseFusedQ4K,
  getQ4KOutputDtype,
  getWeightLayout,
  convertBF16ToF32CPU,
  convertF16ToF32CPU,
  loadQ4KFused,
  loadQ4KDequant,
  loadQ6K,
  loadBF16,
  loadFloat,
  loadTensorToGPU,
  loadTensorToCPU,
} from './tensors/tensor-loader.js';

export { assembleShardData } from './tensors/tensor-reader.js';

export { getTensorNamesByRole } from './tensors/tensor-role.js';

// ============================================================================
// Weight Stage Loaders
// ============================================================================

export { loadEmbeddings } from './embedding-loader.js';

export { loadLayer } from './layer-loader.js';

export { loadFinalWeights } from './final-weights-loader.js';

// ============================================================================
// Utilities
// ============================================================================

export {
  f16ToF32,
  convertBF16ToF32GPU,
  shouldDequantizeToF16,
  applyBufferLayout,
} from './dtype-utils.js';

export {
  maybeDowncastToF16,
  batchDowncastWeights,
} from './weight-downcast.js';

export {
  captureMemorySnapshot,
  formatMemoryStats,
  MemoryMonitor,
  MemoryTimeSeries,
} from './memory-monitor.js';

export {
  QK_K,
  QK4_K_BLOCK_SIZE,
  K_SCALE_SIZE,
  Q4K_BLOCK_BYTES,
  Q6K_BLOCK_BYTES,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_SIZE,
  padToQ4KBlock,
  q4kBlockCount,
} from './quantization-constants.js';

// ============================================================================
// Type-only modules (re-exported for barrel completeness)
// ============================================================================

export {} from './loader-types.js';
export {} from './weights.js';
