// Platform schema contracts.
export {
  // Types
  type PlatformDetectionPattern,
  type PlatformDetectionSchema,
  type MemoryHintsSchema,
  type PlatformSchema,
  type RuntimeCapabilities,
  type ResolvedPlatformConfig,
  type PlatformRegistrySchema,
} from "./platform.schema.js";

export {
  // Types
  type GpuCacheConfigSchema,

  // Defaults
  DEFAULT_GPU_CACHE_CONFIG,
} from "./gpu-cache.schema.js";

export {
  // Types
  type HeapTestingConfigSchema,
  type SegmentTestingConfigSchema,
  type AddressSpaceConfigSchema,
  type SegmentAllocationConfigSchema,
  type EmulatedStorageConfigSchema,
  type MemoryLimitsConfigSchema,

  // Defaults
  DEFAULT_HEAP_TESTING_CONFIG,
  DEFAULT_SEGMENT_TESTING_CONFIG,
  DEFAULT_ADDRESS_SPACE_CONFIG,
  DEFAULT_SEGMENT_ALLOCATION_CONFIG,
  DEFAULT_EMULATED_STORAGE_CONFIG,
  DEFAULT_MEMORY_LIMITS_CONFIG,
} from "./memory-limits.schema.js";

export {
  // Types
  type EmulatedChipType,
  type EmulationTimingMode,
  type EmulatedGPUSpec,
  type EmulatedCPUSpec,
  type NVLinkSpec,
  type NVLinkC2CSpec,
  type EmulatedClusterTopology,
  type TensorParallelConfig,
  type PipelineParallelConfig,
  type DataParallelConfig,
  type ExpertParallelConfig,
  type EmulatedParallelismConfig,
  type EmulatedTimingScaling,
  type LocalResourceTier,
  type LocalResourceMapping,
  type EmulationConfigSchema,
  type VirtualGPUStats,
  type NVLinkStats,
  type EmulationStats,

  // Defaults
  DEFAULT_GH200_GPU_SPEC,
  DEFAULT_GH200_CPU_SPEC,
  DEFAULT_NVLINK_SPEC,
  DEFAULT_NVLINK_C2C_SPEC,
  DEFAULT_PARALLELISM_CONFIG,
  DEFAULT_EMULATION_CONFIG,

  // Factory
  createEmulationConfig,
  getChipProfile,
} from "./emulation.schema.js";
