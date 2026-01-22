
const loadJson = async (path) => {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Failed to load preset: ${path}`);
  return response.json();
};

const gh200Preset = await loadJson('../presets/platforms/nvidia-gh200.json');
const gh200Nvl2Preset = await loadJson('../presets/platforms/nvidia-gh200-nvl2.json');
const gb2008Preset = await loadJson('../presets/platforms/nvidia-gb200-8gpu.json');
const gb200Nvl72Preset = await loadJson('../presets/platforms/nvidia-gb200-nvl72.json');
import { GB } from './units.schema.js';

// =============================================================================
// GPU Specifications
// =============================================================================

export const H100_GPU_SPEC = {
  name: 'H100',
  vramBytes: 96 * GB,
  hbmBandwidthBytesPerSec: 3.35e12,   // 3.35 TB/s HBM3
  fp16Tflops: 1979,                    // ~2 PFLOPS
};

export const H200_GPU_SPEC = {
  name: 'H200',
  vramBytes: 144 * GB,
  hbmBandwidthBytesPerSec: 4.8e12,     // 4.8 TB/s HBM3e
  fp16Tflops: 1979,                     // ~2 PFLOPS
};

export const B200_GPU_SPEC = {
  name: 'B200',
  vramBytes: 192 * GB,
  hbmBandwidthBytesPerSec: 8e12,       // 8 TB/s HBM3e
  fp16Tflops: 4500,                     // 4.5 PFLOPS
  fp8Tflops: 9000,                      // 9 PFLOPS FP8
};

export const DEFAULT_GH200_GPU_SPEC = H200_GPU_SPEC;

// =============================================================================
// CPU Specifications
// =============================================================================

export const GRACE_CPU_SPEC = {
  name: 'Grace',
  cores: 72,
  memoryBytes: 480 * GB, // LPDDR5X
  memoryBandwidthBytesPerSec: 546e9,      // 546 GB/s
};

export const DEFAULT_GH200_CPU_SPEC = GRACE_CPU_SPEC;

// =============================================================================
// NVLink Specifications
// =============================================================================

export const NVLINK_4_SPEC = {
  bandwidthBytesPerSec: 900e9, // 900 GB/s
  latencyUs: 1.0,              // ~1 microsecond
};

export const NVLINK_5_SPEC = {
  bandwidthBytesPerSec: 1.8e12, // 1.8 TB/s
  latencyUs: 0.8,               // ~0.8 microseconds
};

export const DEFAULT_NVLINK_SPEC = NVLINK_4_SPEC;

export const DEFAULT_NVLINK_C2C_SPEC = {
  bandwidthBytesPerSec: 900e9, // 900 GB/s
  latencyUs: 0.5,              // Lower latency for coherent access
  coherent: true,
};

// =============================================================================
// Cluster Topologies
// =============================================================================

export const GH200_TOPOLOGY = {
  gpuCount: 1,
  gpusPerNode: 1,
  nodeCount: 1,
  cpuCount: 1,
};

export const GH200_NVL2_TOPOLOGY = {
  gpuCount: 2,
  gpusPerNode: 2,
  nodeCount: 1,
  cpuCount: 2,
};

export const GB200_8GPU_TOPOLOGY = {
  gpuCount: 8,
  gpusPerNode: 8,
  nodeCount: 1,
  cpuCount: 2,
};

export const GB200_NVL72_TOPOLOGY = {
  gpuCount: 72,
  gpusPerNode: 8,
  nodeCount: 9,
  cpuCount: 18, // 2 CPUs per node
};

// =============================================================================
// Parallelism Defaults
// =============================================================================

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

export const TP2_PARALLELISM_CONFIG = {
  ...DEFAULT_PARALLELISM_CONFIG,
  tensorParallel: {
    enabled: true,
    degree: 2,
  },
};

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

export const DEFAULT_TIMING_SCALING = {
  computeScale: 1.0,
  memoryScale: 1.0,
  nvlinkScale: 1.0,
};

// =============================================================================
// Complete Emulation Config Defaults
// =============================================================================

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
  maxActiveWorkingSetBytes: 4 * GB, // default working set
  statsEnabled: true,
  logOperations: false,
};

// =============================================================================
// Chip Presets
// =============================================================================

const CHIP_PRESETS = {
  'gh200': gh200Preset.emulation,
  'gh200-nvl2': gh200Nvl2Preset.emulation,
  'gb200-8gpu': gb2008Preset.emulation,
  'gb200-nvl72': gb200Nvl72Preset.emulation,
};

export function getChipPreset(chipType) {
  const preset = CHIP_PRESETS[chipType];
  if (!preset) {
    throw new Error(`Unknown chip type: ${chipType}. Valid types: ${Object.keys(CHIP_PRESETS).join(', ')}`);
  }
  return { ...preset };
}

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

export function calculateTotalVram(config) {
  return config.gpuSpec.vramBytes * config.topology.gpuCount;
}

export function calculateTotalCpuMemory(config) {
  return config.cpuSpec.memoryBytes * config.topology.cpuCount;
}

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

export function formatBandwidth(bytesPerSec) {
  return `${formatBytes(bytesPerSec)}/s`;
}
