// Platform schema contracts.
export {
  DEFAULT_GPU_CACHE_CONFIG,
} from "./gpu-cache.schema.js";

export {
  DEFAULT_HEAP_TESTING_CONFIG,
  DEFAULT_SEGMENT_TESTING_CONFIG,
  DEFAULT_ADDRESS_SPACE_CONFIG,
  DEFAULT_SEGMENT_ALLOCATION_CONFIG,
  DEFAULT_EMULATED_STORAGE_CONFIG,
  DEFAULT_MEMORY_LIMITS_CONFIG,
} from "./memory-limits.schema.js";

export {
  // Factory
  createEmulationConfig,
  getChipProfile,

  // Defaults
  DEFAULT_GH200_GPU_SPEC,
  DEFAULT_GH200_CPU_SPEC,
  DEFAULT_NVLINK_SPEC,
  DEFAULT_NVLINK_C2C_SPEC,
  DEFAULT_PARALLELISM_CONFIG,
  DEFAULT_EMULATION_CONFIG,
} from "./emulation.schema.js";
