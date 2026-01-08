/**
 * Schema Index
 *
 * Re-exports all schema definitions for easy importing.
 *
 * Naming Convention:
 * - *Schema: Type definitions (interface structure)
 * - *Config: Runtime instances (validated values)
 * - *Raw: Unparsed input (from manifest/file)
 * - *Options: Function parameters
 *
 * @module config/schema
 */

// =============================================================================
// Manifest Schema
// =============================================================================
export {
  // Constants
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,

  // Defaults
  DEFAULT_MANIFEST_INFERENCE,

  // Helpers
  isV1Manifest,
  hasMoEConfig,
  validateManifestInference,
  hasInferenceConfig,
} from './manifest.schema.js';

// =============================================================================
// Kernel Path Schema
// =============================================================================
export {
  DEFAULT_ENTRY,
  DEFAULT_INPUT,
  DEFAULT_OUTPUT,
} from './kernel-path.schema.js';

// =============================================================================
// Inference Schema
// =============================================================================
export {
  // RoPE
  DEFAULT_ROPE_CONFIG,

  // Architecture defaults
  DEFAULT_MAX_POSITION_EMBEDDINGS,

  // Functions
  computeGlobalLayers,
} from './inference.schema.js';

// =============================================================================
// Conversion Schema
// =============================================================================
export {
  // Constants
  ConversionStage,
} from './conversion.schema.js';

// =============================================================================
// Loading Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_SHARD_CACHE_CONFIG,
  DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  DEFAULT_OPFS_PATH_CONFIG,
  DEFAULT_EXPERT_CACHE_CONFIG,
  DEFAULT_LOADING_CONFIG,
} from './loading.schema.js';

// =============================================================================
// Kernel Registry Schema
// =============================================================================
export {
  // Functions
  mergeBindings,
  resolveKernelConfig,
} from './kernel-registry.schema.js';

// =============================================================================
// Storage Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_QUOTA_CONFIG,
  DEFAULT_VRAM_ESTIMATION_CONFIG,
  DEFAULT_STORAGE_ALIGNMENT_CONFIG,
  DEFAULT_STORAGE_FULL_CONFIG,
} from './storage.schema.js';

// =============================================================================
// Inference Defaults Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_BATCHING_DEFAULTS,
  DEFAULT_COMPUTE_DEFAULTS,
  DEFAULT_LARGE_WEIGHT_CONFIG,
  DEFAULT_SAMPLING_DEFAULTS,
  DEFAULT_TOKENIZER_DEFAULTS,
  DEFAULT_INFERENCE_DEFAULTS_CONFIG,
} from './inference-defaults.schema.js';

// =============================================================================
// Distribution Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_DISTRIBUTION_CONFIG,
} from './distribution.schema.js';

// =============================================================================
// MoE Runtime Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_MOE_ROUTING_CONFIG,
  DEFAULT_MOE_CACHE_CONFIG,
  DEFAULT_MOE_RUNTIME_CONFIG,
} from './moe.schema.js';

// =============================================================================
// KV Cache Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_KVCACHE_CONFIG,

  // Thresholds
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} from './kvcache.schema.js';

// =============================================================================
// GPU Cache Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_GPU_CACHE_CONFIG,
} from './gpu-cache.schema.js';

// =============================================================================
// Tuner Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_TUNER_CONFIG,
} from './tuner.schema.js';

// =============================================================================
// Debug Schema
// =============================================================================
export {
  // Constants
  LOG_LEVELS,

  // Defaults
  DEFAULT_LOG_OUTPUT_CONFIG,
  DEFAULT_LOG_HISTORY_CONFIG,
  DEFAULT_LOG_LEVEL_CONFIG,
  DEFAULT_TRACE_CONFIG,
  DEFAULT_PIPELINE_DEBUG_CONFIG,
  DEFAULT_DEBUG_CONFIG,
} from './debug.schema.js';

// =============================================================================
// Hot-Swap Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_HOTSWAP_CONFIG,
} from './hotswap.schema.js';

// =============================================================================
// Buffer Pool Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
  DEFAULT_BUFFER_POOL_CONFIG,
} from './buffer-pool.schema.js';

// =============================================================================
// Memory Limits Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_HEAP_TESTING_CONFIG,
  DEFAULT_SEGMENT_TESTING_CONFIG,
  DEFAULT_ADDRESS_SPACE_CONFIG,
  DEFAULT_SEGMENT_ALLOCATION_CONFIG,
  DEFAULT_MEMORY_LIMITS_CONFIG,
} from './memory-limits.schema.js';

// =============================================================================
// Bridge Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_BRIDGE_CONFIG,
} from './bridge.schema.js';

// =============================================================================
// Quantization Defaults Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_QUANTIZATION_DEFAULTS,
} from './quantization-defaults.schema.js';

// =============================================================================
// Kernel Thresholds Schema
// =============================================================================
export {
  // Constants
  DTYPE_SIZES,

  // Defaults
  DEFAULT_MATMUL_THRESHOLDS,
  DEFAULT_RMSNORM_THRESHOLDS,
  DEFAULT_ROPE_DEFAULTS,
  DEFAULT_ATTENTION_THRESHOLDS,
  DEFAULT_CAST_THRESHOLDS,
  DEFAULT_KERNEL_THRESHOLDS,

  // Functions
  getKernelThresholds,
  setKernelThresholds,
  resetKernelThresholds,
} from './kernel-thresholds.schema.js';

// =============================================================================
// Doppler Master Config
// =============================================================================
export {
  // Defaults
  DEFAULT_RUNTIME_CONFIG,
  DEFAULT_DOPPLER_CONFIG,

  // Factory
  createDopplerConfig,
} from './doppler.schema.js';
