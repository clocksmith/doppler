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

  // Types
  type HashAlgorithm,
  type ModelType,
  type ComponentGroupType,
  type WeightLayout,
  type QuantizationValue,

  // Schemas
  type ArchitectureSchema,
  type ShardSchema,
  type TensorSpanSchema,
  type TensorSchema,
  type TensorMapSchema,
  type ComponentGroupSchema,
  type MoEConfigSchema,
  type TokenizerSchema,
  type KernelHintsSchema,
  type RuntimeOptimizationsSchema,
  type QuantizationInfoSchema,
  type ConversionInfoSchema,
  type ManifestSchema,

  // Helpers
  isV1Manifest,
  hasMoEConfig,
} from './manifest.schema.js';

// =============================================================================
// Inference Schema
// =============================================================================
export {
  // RoPE
  type RoPEConfigSchema,
  DEFAULT_ROPE_CONFIG,

  type AttentionSchema,
  type NormalizationSchema,
  type FFNSchema,
  type LayerPipelineOp,
  type LayerPipelineNormWeight,
  type LayerPipelineStepSchema,
  type LayerPipelineOverrideSchema,
  type LayerPipelineSchema,
  type OutputSchema,
  type LayerType,
  type GlobalLayerPattern,
  type LayerPatternSchema,
  type InferenceConfigSchema,
  type SamplingSchema,
  type TokenizerConfigSchema,
  // Functions
  computeGlobalLayers,
} from './inference.schema.js';

// =============================================================================
// Conversion Schema
// =============================================================================
export {
  // Types
  type QuantizationType,
  type ConversionStageType,

  // Constants
  ConversionStage,

  // Schemas
  type TensorInfoSchema,
  type ParsedModelSchema,
  type RawModelConfigSchema,
  type ConversionOptionsSchema,
  type ConversionProgressSchema,
  type WriterOptionsSchema,
  type TensorLocationSchema,
  type WriteResultSchema,
  type ConversionIOSchema,
} from './conversion.schema.js';

// =============================================================================
// Preset Schema
// =============================================================================
export {
  type PresetSchema,
  type TensorPatternSchema,
  type DetectionPatternSchema,
  type ResolvedConfigSchema,
} from './preset.schema.js';

// =============================================================================
// Loading Schema
// =============================================================================
export {
  // Types
  type ShardCacheConfigSchema,
  type MemoryManagementConfigSchema,
  type OpfsPathConfigSchema,
  type ExpertCacheConfigSchema,
  type LoadingConfigSchema,

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
  // Types
  type GpuFeature,
  type BindingType,
  type BindingSchema,
  type UniformFieldType,
  type UniformFieldSchema,
  type UniformsSchema,
  type WgslOverridesSchema,
  type KernelVariantSchema,
  type OperationSchema,
  type KernelRegistrySchema,
  type ResolvedKernelConfig,

  // Functions
  mergeBindings,
  resolveKernelConfig,
} from './kernel-registry.schema.js';

// =============================================================================
// Platform Schema
// =============================================================================
export {
  // Types
  type PlatformDetectionSchema,
  type KernelOperationOverrideSchema,
  type KernelOverridesSchema,
  type MemoryHintsSchema,
  type PlatformSchema,
  type RuntimeCapabilities,
  type ResolvedPlatformConfig,
  type PlatformRegistrySchema,
} from './platform.schema.js';

// =============================================================================
// Storage Schema
// =============================================================================
export {
  // Types
  type QuotaConfigSchema,
  type VramEstimationConfigSchema,
  type StorageAlignmentConfigSchema,
  type StorageFullConfigSchema,

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
  // Types
  type BatchingDefaultsSchema,
  type SamplingDefaultsSchema,
  type TokenizerDefaultsSchema,
  type InferenceDefaultsConfigSchema,

  // Defaults
  DEFAULT_BATCHING_DEFAULTS,
  DEFAULT_SAMPLING_DEFAULTS,
  DEFAULT_TOKENIZER_DEFAULTS,
  DEFAULT_INFERENCE_DEFAULTS_CONFIG,
} from './inference-defaults.schema.js';

// =============================================================================
// Distribution Schema
// =============================================================================
export {
  // Types
  type DistributionConfigSchema,

  // Defaults
  DEFAULT_DISTRIBUTION_CONFIG,
} from './distribution.schema.js';

// =============================================================================
// MoE Runtime Schema
// =============================================================================
export {
  // Types
  type RouterDtype,
  type MoERoutingConfigSchema,
  type MoECacheConfigSchema,
  type MoERuntimeConfigSchema,

  // Defaults
  DEFAULT_MOE_ROUTING_CONFIG,
  DEFAULT_MOE_CACHE_CONFIG,
  DEFAULT_MOE_RUNTIME_CONFIG,
} from './moe.schema.js';

// =============================================================================
// KV Cache Schema
// =============================================================================
export {
  // Types
  type KVDtype,
  type KVLayout,
  type KVCacheConfigSchema,

  // Defaults
  DEFAULT_KVCACHE_CONFIG,
} from './kvcache.schema.js';

// =============================================================================
// GPU Cache Schema
// =============================================================================
export {
  // Types
  type GpuCacheConfigSchema,

  // Defaults
  DEFAULT_GPU_CACHE_CONFIG,
} from './gpu-cache.schema.js';

// =============================================================================
// Tuner Schema
// =============================================================================
export {
  // Types
  type TunerConfigSchema,

  // Defaults
  DEFAULT_TUNER_CONFIG,
} from './tuner.schema.js';

// =============================================================================
// Debug Schema
// =============================================================================
export {
  // Types
  type LogOutputConfigSchema,
  type LogHistoryConfigSchema,
  type LogLevelConfigSchema,
  type TraceCategory,
  type TraceConfigSchema,
  type PipelineDebugCategory,
  type PipelineDebugConfigSchema,
  type ProbeStage,
  type ProbeConfigSchema,
  type DebugConfigSchema,

  // Defaults
  DEFAULT_LOG_OUTPUT_CONFIG,
  DEFAULT_LOG_HISTORY_CONFIG,
  DEFAULT_LOG_LEVEL_CONFIG,
  DEFAULT_TRACE_CONFIG,
  DEFAULT_PIPELINE_DEBUG_CONFIG,
  DEFAULT_DEBUG_CONFIG,
} from './debug.schema.js';

// =============================================================================
// Buffer Pool Schema
// =============================================================================
export {
  // Types
  type BufferPoolBucketConfigSchema,
  type BufferPoolLimitsConfigSchema,
  type BufferPoolAlignmentConfigSchema,
  type BufferPoolConfigSchema,

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
  // Types
  type HeapTestingConfigSchema,
  type SegmentTestingConfigSchema,
  type AddressSpaceConfigSchema,
  type SegmentAllocationConfigSchema,
  type MemoryLimitsConfigSchema,

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
  // Types
  type BridgeConfigSchema,

  // Defaults
  DEFAULT_BRIDGE_CONFIG,
} from './bridge.schema.js';

// =============================================================================
// Doppler Master Config
// =============================================================================
export {
  // Types
  type RuntimeConfigSchema,
  type DopplerConfigSchema,

  // Defaults
  DEFAULT_RUNTIME_CONFIG,
  DEFAULT_DOPPLER_CONFIG,

  // Factory
  createDopplerConfig,
} from './doppler.schema.js';
