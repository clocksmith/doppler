// =============================================================================
// Manifest Schema
// =============================================================================
export {
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,
  MAX_HEADER_SIZE,
  HEADER_READ_SIZE,
  DEFAULT_RMS_NORM_EPS,
  DEFAULT_HIGH_PRECISION_EPS,
  DEFAULT_MANIFEST_INFERENCE,
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
  computeGlobalLayers,
} from './inference.schema.js';

// =============================================================================
// Conversion Schema
// =============================================================================
export {
  ConversionStage,
} from './conversion.schema.js';

// =============================================================================
// Converter Schema
// =============================================================================
export {
  DEFAULT_CONVERTER_QUANTIZATION_CONFIG,
  DEFAULT_CONVERTER_SHARDING_CONFIG,
  DEFAULT_CONVERTER_STREAMING_CONFIG,
  DEFAULT_CONVERTER_HTTP_CONFIG,
  DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG,
  DEFAULT_CONVERTER_MANIFEST_CONFIG,
  DEFAULT_CONVERTER_OUTPUT_CONFIG,
  DEFAULT_CONVERTER_PRESET_CONFIG,
  DEFAULT_CONVERTER_CONFIG,
  createConverterConfig,
} from './converter.schema.js';

// =============================================================================
// Loading Schema
// =============================================================================
export {
  DEFAULT_Q4K_LAYOUT,
  DEFAULT_SHARD_CACHE_CONFIG,
  DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  DEFAULT_PREFETCH_CONFIG,
  DEFAULT_OPFS_PATH_CONFIG,
  DEFAULT_EXPERT_CACHE_CONFIG,
  DEFAULT_LOADING_CONFIG,
} from './loading.schema.js';

// =============================================================================
// Kernel Registry Schema
// =============================================================================
export {
  mergeBindings,
  resolveKernelConfig,
} from './kernel-registry.schema.js';

// =============================================================================
// Storage Schema
// =============================================================================
export {
  DEFAULT_QUOTA_CONFIG,
  DEFAULT_VRAM_ESTIMATION_CONFIG,
  DEFAULT_STORAGE_ALIGNMENT_CONFIG,
  DEFAULT_STORAGE_FULL_CONFIG,
} from './storage.schema.js';

// =============================================================================
// Inference Defaults Schema
// =============================================================================
export {
  DEFAULT_GENERATION_CONFIG,
  DEFAULT_BATCHING_DEFAULTS,
  DEFAULT_COMPUTE_DEFAULTS,
  DEFAULT_LARGE_WEIGHT_CONFIG,
  DEFAULT_SAMPLING_DEFAULTS,
  DEFAULT_TOKENIZER_DEFAULTS,
  DEFAULT_CHAT_TEMPLATE_CONFIG,
  DEFAULT_INFERENCE_DEFAULTS_CONFIG,
  DEFAULT_PRESET_INFERENCE_CONFIG,
} from './inference-defaults.schema.js';

// =============================================================================
// Distribution Schema
// =============================================================================
export {
  DEFAULT_DISTRIBUTION_CONFIG,
} from './distribution.schema.js';

// =============================================================================
// MoE Runtime Schema
// =============================================================================
export {
  DEFAULT_MOE_ROUTING_CONFIG,
  DEFAULT_MOE_CACHE_CONFIG,
  DEFAULT_MOE_RUNTIME_CONFIG,
} from './moe.schema.js';

// =============================================================================
// KV Cache Schema
// =============================================================================
export {
  DEFAULT_KVCACHE_CONFIG,
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} from './kvcache.schema.js';

// =============================================================================
// GPU Cache Schema
// =============================================================================
export {
  DEFAULT_GPU_CACHE_CONFIG,
} from './gpu-cache.schema.js';

// =============================================================================
// Tuner Schema
// =============================================================================
export {
  DEFAULT_TUNER_CONFIG,
} from './tuner.schema.js';

// =============================================================================
// Debug Schema
// =============================================================================
export {
  LOG_LEVELS,
  DEFAULT_LOG_OUTPUT_CONFIG,
  DEFAULT_LOG_HISTORY_CONFIG,
  DEFAULT_LOG_LEVEL_CONFIG,
  DEFAULT_TRACE_CONFIG,
  DEFAULT_KERNEL_TRACE_CONFIG,
  DEFAULT_PIPELINE_DEBUG_CONFIG,
  DEFAULT_PROFILER_CONFIG,
  DEFAULT_PERF_GUARDS_CONFIG,
  DEFAULT_DEBUG_CONFIG,
} from './debug.schema.js';

// =============================================================================
// Benchmark Schema
// =============================================================================
export {
  DEFAULT_BENCHMARK_OUTPUT_CONFIG,
  DEFAULT_BENCHMARK_RUN_CONFIG,
  DEFAULT_BENCHMARK_STATS_CONFIG,
  DEFAULT_BENCHMARK_COMPARISON_CONFIG,
  DEFAULT_BENCHMARK_BASELINE_CONFIG,
  DEFAULT_BENCHMARK_CONFIG,
} from './benchmark.schema.js';

// =============================================================================
// Tooling Schema
// =============================================================================
export {
  TOOLING_INTENTS,
  TOOLING_DIAGNOSTICS,
  DEFAULT_TOOLING_CONFIG,
} from './tooling.schema.js';

// =============================================================================
// Hot-Swap Schema
// =============================================================================
export {
  DEFAULT_HOTSWAP_CONFIG,
} from './hotswap.schema.js';

// =============================================================================
// Buffer Pool Schema
// =============================================================================
export {
  DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
  DEFAULT_BUFFER_POOL_CONFIG,
} from './buffer-pool.schema.js';

// =============================================================================
// Memory Limits Schema
// =============================================================================
export {
  DEFAULT_HEAP_TESTING_CONFIG,
  DEFAULT_SEGMENT_TESTING_CONFIG,
  DEFAULT_ADDRESS_SPACE_CONFIG,
  DEFAULT_SEGMENT_ALLOCATION_CONFIG,
  DEFAULT_EMULATED_STORAGE_CONFIG,
  DEFAULT_MEMORY_LIMITS_CONFIG,
} from './memory-limits.schema.js';

// =============================================================================
// Bridge Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_BRIDGE_TIMEOUT_CONFIG,
  DEFAULT_BRIDGE_CONFIG,
} from './bridge.schema.js';

// =============================================================================
// Adapter Schema
// =============================================================================
export {
  // Constants
  VALID_LORA_TARGET_MODULES,

  // Defaults
  DEFAULT_ADAPTER_VALIDATION_CONFIG,
  DEFAULT_ADAPTER_STACK_CONFIG,
  DEFAULT_ADAPTER_REGISTRY_CONFIG,
  DEFAULT_ADAPTER_CONFIG,
} from './adapter.schema.js';

// =============================================================================
// LoRA Schema
// =============================================================================
export {
  DEFAULT_LORA_CONFIG,
} from './lora.schema.js';

// =============================================================================
// Training Schema
// =============================================================================
export {
  DEFAULT_TRAINING_OPTIMIZER_CONFIG,
  DEFAULT_TRAINING_GRADIENT_CONFIG,
  DEFAULT_TRAINING_PRECISION_CONFIG,
  DEFAULT_TRAINING_ATTENTION_CONFIG,
  DEFAULT_TRAINING_LOSS_SCALING_CONFIG,
  DEFAULT_TRAINING_SETTINGS,
} from './training.schema.js';

// =============================================================================
// Backward Registry Schema
// =============================================================================
export {
  validateBackwardRegistry,
} from './backward-registry.schema.js';

// =============================================================================
// Quantization Defaults Schema
// =============================================================================
export {
  // Defaults
  DEFAULT_QUANTIZATION_DEFAULTS,
} from './quantization-defaults.schema.js';

// =============================================================================
// Quantization Constants (Invariants)
// =============================================================================
export {
  // Constants
  QK_K,
  Q4K_BLOCK_BYTES,
  Q6K_BLOCK_BYTES,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_SIZE,
  K_SCALE_SIZE,
  QK4_K_BLOCK_SIZE,

  // Functions
  padToQ4KBlock,
  q4kBlockCount,
} from './quantization.schema.js';

// =============================================================================
// Unit Constants (Invariants)
// =============================================================================
export {
  KB,
  MB,
  GB,
  formatBytes,
  formatBytesCompact,
} from './units.schema.js';

// =============================================================================
// Kernel Thresholds Schema
// =============================================================================
export {
  // Constants
  DTYPE_SIZES,
  getDtypeSize,

  // Defaults
  DEFAULT_MATMUL_THRESHOLDS,
  DEFAULT_RMSNORM_THRESHOLDS,
  DEFAULT_SOFTMAX_THRESHOLDS,
  DEFAULT_FFN_THRESHOLDS,
  DEFAULT_SAMPLE_THRESHOLDS,
  DEFAULT_ROPE_DEFAULTS,
  DEFAULT_ATTENTION_THRESHOLDS,
  DEFAULT_CAST_THRESHOLDS,
  DEFAULT_TUNER_LIMITS,
  DEFAULT_KERNEL_THRESHOLDS,

  // Functions
  getKernelThresholds,
  setKernelThresholds,
  resetKernelThresholds,
} from './kernel-thresholds.schema.js';

// =============================================================================
// Emulation Schema
// =============================================================================
export {
  // Factory
  createEmulationConfig,
  getChipPreset,

  // Defaults
  DEFAULT_GH200_GPU_SPEC,
  DEFAULT_GH200_CPU_SPEC,
  DEFAULT_NVLINK_SPEC,
  DEFAULT_NVLINK_C2C_SPEC,
  DEFAULT_PARALLELISM_CONFIG,
  DEFAULT_EMULATION_CONFIG,
} from './emulation.schema.js';

// =============================================================================
// Shared Runtime Schema
// =============================================================================
export {
  DEFAULT_KERNEL_REGISTRY_CONFIG,
  DEFAULT_SHARED_RUNTIME_CONFIG,
} from './shared-runtime.schema.js';
export {
  DEFAULT_HARNESS_CONFIG,
} from './harness.schema.js';
export {
  DEFAULT_SPECULATIVE_CONFIG,
} from './speculative.schema.js';

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
