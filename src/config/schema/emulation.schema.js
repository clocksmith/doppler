/**
 * Emulation Config Schema - Defaults and Factory
 *
 * Configuration for NVIDIA superchip emulation using local resources.
 *
 * @module config/schema/emulation
 */

import gh200Preset from '../presets/platforms/nvidia-gh200.json' with { type: 'json' };
import gh200Nvl2Preset from '../presets/platforms/nvidia-gh200-nvl2.json' with { type: 'json' };
import gb2008Preset from '../presets/platforms/nvidia-gb200-8gpu.json' with { type: 'json' };
import gb200Nvl72Preset from '../presets/platforms/nvidia-gb200-nvl72.json' with { type: 'json' };

// =============================================================================
// GPU Specifications
// =============================================================================

/** H100 GPU spec (96GB HBM3) */
export const H100_GPU_SPEC = {
  name: 'H100',
  vramBytes: 96 * 1024 * 1024 * 1024, // 96GB
  hbmBandwidthBytesPerSec: 3.35e12,   // 3.35 TB/s HBM3
  fp16Tflops: 1979,                    // ~2 PFLOPS
};

/** H200 GPU spec (144GB HBM3e) */
export const H200_GPU_SPEC = {
  name: 'H200',
  vramBytes: 144 * 1024 * 1024 * 1024, // 144GB
  hbmBandwidthBytesPerSec: 4.8e12,     // 4.8 TB/s HBM3e
  fp16Tflops: 1979,                     // ~2 PFLOPS
};

/** B200 GPU spec (192GB HBM3e) */
export const B200_GPU_SPEC = {
  name: 'B200',
  vramBytes: 192 * 1024 * 1024 * 1024, // 192GB
  hbmBandwidthBytesPerSec: 8e12,       // 8 TB/s HBM3e
  fp16Tflops: 4500,                     // 4.5 PFLOPS
  fp8Tflops: 9000,                      // 9 PFLOPS FP8
};

/** Default GH200 GPU spec (H200 variant) */
export const DEFAULT_GH200_GPU_SPEC = H200_GPU_SPEC;

// =============================================================================
// CPU Specifications
// =============================================================================

/** Grace CPU spec */
export const GRACE_CPU_SPEC = {
  name: 'Grace',
  cores: 72,
  memoryBytes: 480 * 1024 * 1024 * 1024, // 480GB LPDDR5X
  memoryBandwidthBytesPerSec: 546e9,      // 546 GB/s
};

/** Default GH200 CPU spec (Grace) */
export const DEFAULT_GH200_CPU_SPEC = GRACE_CPU_SPEC;

// =============================================================================
// NVLink Specifications
// =============================================================================

/** NVLink 4.0 spec (GH200) - 900 GB/s */
export const NVLINK_4_SPEC = {
  bandwidthBytesPerSec: 900e9, // 900 GB/s
  latencyUs: 1.0,              // ~1 microsecond
};

/** NVLink 5.0 spec (GB200) - 1.8 TB/s */
export const NVLINK_5_SPEC = {
  bandwidthBytesPerSec: 1.8e12, // 1.8 TB/s
  latencyUs: 0.8,               // ~0.8 microseconds
};

/** Default NVLink spec (GH200 - 900 GB/s) */
export const DEFAULT_NVLINK_SPEC = NVLINK_4_SPEC;

/** NVLink-C2C spec (CPU↔GPU coherent) */
export const DEFAULT_NVLINK_C2C_SPEC = {
  bandwidthBytesPerSec: 900e9, // 900 GB/s
  latencyUs: 0.5,              // Lower latency for coherent access
  coherent: true,
};

// =============================================================================
// Cluster Topologies
// =============================================================================

/** GH200 single superchip topology */
export const GH200_TOPOLOGY = {
  gpuCount: 1,
  gpusPerNode: 1,
  nodeCount: 1,
  cpuCount: 1,
};

/** GH200 NVL2 topology (2 superchips) */
export const GH200_NVL2_TOPOLOGY = {
  gpuCount: 2,
  gpusPerNode: 2,
  nodeCount: 1,
  cpuCount: 2,
};

/** GB200 8-GPU Pod topology */
export const GB200_8GPU_TOPOLOGY = {
  gpuCount: 8,
  gpusPerNode: 8,
  nodeCount: 1,
  cpuCount: 2,
};

/** GB200 NVL72 topology (72 GPUs, 9 nodes) */
export const GB200_NVL72_TOPOLOGY = {
  gpuCount: 72,
  gpusPerNode: 8,
  nodeCount: 9,
  cpuCount: 18, // 2 CPUs per node
};

// =============================================================================
// Parallelism Defaults
// =============================================================================

/** Default parallelism configuration (no parallelism) */
export const DEFAULT_PARALLELISM_CONFIG = {
  tensorParallel: {
    enabled: false,
    degree: 1,
  },
  pipelineParallel: {
    enabled: false,
    stages: 1,
    microBatches: 1,
  },
  dataParallel: {
    enabled: false,
    degree: 1,
  },
  expertParallel: {
    enabled: false,
    degree: 1,
  },
};

/** TP=2 parallelism for GH200 NVL2 */
export const TP2_PARALLELISM_CONFIG = {
  ...DEFAULT_PARALLELISM_CONFIG,
  tensorParallel: {
    enabled: true,
    degree: 2,
  },
};

/** TP=8 parallelism for GB200 8-GPU */
export const TP8_PARALLELISM_CONFIG = {
  ...DEFAULT_PARALLELISM_CONFIG,
  tensorParallel: {
    enabled: true,
    degree: 8,
  },
};

// =============================================================================
// Timing Scaling Defaults
// =============================================================================

/** Default timing scaling (no scaling) */
export const DEFAULT_TIMING_SCALING = {
  computeScale: 1.0,
  memoryScale: 1.0,
  nvlinkScale: 1.0,
};

// =============================================================================
// Complete Emulation Config Defaults
// =============================================================================

/** Default emulation configuration (disabled) */
export const DEFAULT_EMULATION_CONFIG = {
  enabled: false,
  targetChip: 'gh200',
  timingMode: 'functional',
  gpuSpec: DEFAULT_GH200_GPU_SPEC,
  cpuSpec: DEFAULT_GH200_CPU_SPEC,
  topology: GH200_TOPOLOGY,
  nvlink: DEFAULT_NVLINK_SPEC,
  nvlinkC2C: DEFAULT_NVLINK_C2C_SPEC,
  parallelism: DEFAULT_PARALLELISM_CONFIG,
  timingScaling: DEFAULT_TIMING_SCALING,
  localResources: undefined,
  opfsRootPath: 'emulation',
  maxActiveWorkingSetBytes: 4 * 1024 * 1024 * 1024, // 4GB default working set
  statsEnabled: true,
  logOperations: false,
};

// =============================================================================
// Chip Presets
// =============================================================================

/** Preset configurations for each chip type */
const CHIP_PRESETS = {
  'gh200': gh200Preset.emulation,
  'gh200-nvl2': gh200Nvl2Preset.emulation,
  'gb200-8gpu': gb2008Preset.emulation,
  'gb200-nvl72': gb200Nvl72Preset.emulation,
};

/**
 * Get preset config for a specific chip type
 * @param {string} chipType - Target chip type
 * @returns {Object} Preset configuration
 */
export function getChipPreset(chipType) {
  const preset = CHIP_PRESETS[chipType];
  if (!preset) {
    throw new Error(`Unknown chip type: ${chipType}. Valid types: ${Object.keys(CHIP_PRESETS).join(', ')}`);
  }
  return { ...preset };
}

/**
 * Create emulation config with overrides
 * @param {Object} overrides - Configuration overrides
 * @returns {Object} Complete emulation config
 */
export function createEmulationConfig(overrides) {
  if (!overrides) {
    return { ...DEFAULT_EMULATION_CONFIG };
  }

  // If targetChip is specified, apply preset first
  const chipPreset = overrides.targetChip
    ? getChipPreset(overrides.targetChip)
    : {};
  const { enabled: _enabled, ...chipPresetConfig } = chipPreset;
  const presetParallelism = chipPresetConfig.parallelism
    ? mergeParallelismConfig(DEFAULT_PARALLELISM_CONFIG, chipPresetConfig.parallelism)
    : DEFAULT_PARALLELISM_CONFIG;
  const resolvedParallelism = overrides.parallelism
    ? mergeParallelismConfig(presetParallelism, overrides.parallelism)
    : presetParallelism;

  return {
    ...DEFAULT_EMULATION_CONFIG,
    ...chipPresetConfig,
    ...overrides,
    enabled: overrides.enabled ?? DEFAULT_EMULATION_CONFIG.enabled,
    // Deep merge nested objects
    gpuSpec: {
      ...DEFAULT_EMULATION_CONFIG.gpuSpec,
      ...chipPresetConfig.gpuSpec,
      ...overrides.gpuSpec,
    },
    cpuSpec: {
      ...DEFAULT_EMULATION_CONFIG.cpuSpec,
      ...chipPresetConfig.cpuSpec,
      ...overrides.cpuSpec,
    },
    topology: {
      ...DEFAULT_EMULATION_CONFIG.topology,
      ...chipPresetConfig.topology,
      ...overrides.topology,
    },
    nvlink: {
      ...DEFAULT_EMULATION_CONFIG.nvlink,
      ...chipPresetConfig.nvlink,
      ...overrides.nvlink,
    },
    nvlinkC2C: {
      ...DEFAULT_EMULATION_CONFIG.nvlinkC2C,
      ...chipPresetConfig.nvlinkC2C,
      ...overrides.nvlinkC2C,
    },
    parallelism: resolvedParallelism,
    timingScaling: {
      ...DEFAULT_EMULATION_CONFIG.timingScaling,
      ...overrides.timingScaling,
    },
  };
}

/**
 * Merge parallelism configuration
 * @param {Object} base - Base config
 * @param {Object} overrides - Override config
 * @returns {Object} Merged config
 */
function mergeParallelismConfig(base, overrides) {
  return {
    tensorParallel: {
      ...base.tensorParallel,
      ...overrides.tensorParallel,
    },
    pipelineParallel: {
      ...base.pipelineParallel,
      ...overrides.pipelineParallel,
    },
    dataParallel: {
      ...base.dataParallel,
      ...overrides.dataParallel,
    },
    expertParallel: {
      ...base.expertParallel,
      ...overrides.expertParallel,
    },
  };
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate total VRAM for a configuration
 * @param {Object} config - Emulation config
 * @returns {number} Total VRAM in bytes
 */
export function calculateTotalVram(config) {
  return config.gpuSpec.vramBytes * config.topology.gpuCount;
}

/**
 * Calculate total CPU memory for a configuration
 * @param {Object} config - Emulation config
 * @returns {number} Total CPU memory in bytes
 */
export function calculateTotalCpuMemory(config) {
  return config.cpuSpec.memoryBytes * config.topology.cpuCount;
}

/**
 * Format bytes as human-readable string
 * @param {number} bytes - Byte count
 * @returns {string} Formatted string (e.g., "144 GB")
 */
export function formatBytes(bytes) {
  if (bytes >= 1e12) {
    return `${(bytes / 1e12).toFixed(1)} TB`;
  } else if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  } else if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  } else if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

/**
 * Format bandwidth as human-readable string
 * @param {number} bytesPerSec - Bandwidth in bytes/sec
 * @returns {string} Formatted string (e.g., "900 GB/s")
 */
export function formatBandwidth(bytesPerSec) {
  return `${formatBytes(bytesPerSec)}/s`;
}
