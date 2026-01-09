// =============================================================================
// Matmul Thresholds
// =============================================================================

export const DEFAULT_MATMUL_THRESHOLDS = {
  multicolThreshold: 256,
};

// =============================================================================
// RMSNorm Thresholds
// =============================================================================

export const DEFAULT_RMSNORM_THRESHOLDS = {
  smallThreshold: 256,
};

// =============================================================================
// RoPE Thresholds
// =============================================================================

export const DEFAULT_ROPE_DEFAULTS = {
  defaultTheta: 10000.0,
  uniformSize: 32,
};

// =============================================================================
// Attention Thresholds
// =============================================================================

export const DEFAULT_ATTENTION_THRESHOLDS = {
  chunkedMaxKVLen: 2048,
  minHeadDimForChunked: 128,
  tierHeadDimLimits: {
    tier3: 64,
    tier2: 128,
    tier1: 256,
  },
  tierMinSharedMemory: {
    tier3: 16384,  // 16KB for small models
    tier2: 32768,  // 32KB for medium models
    tier1: 65536,  // 64KB for large models
  },
};

// =============================================================================
// Fused Matmul Thresholds
// =============================================================================

export const DEFAULT_FUSED_MATMUL_THRESHOLDS = {
  maxMediumN: 4096,
  colsPerWg: 4,
};

// =============================================================================
// Cast Thresholds
// =============================================================================

export const DEFAULT_CAST_THRESHOLDS = {
  maxWorkgroupsPerDim: 65535,
};

// =============================================================================
// Dtype Size Constants
// =============================================================================

export const DTYPE_SIZES = {
  f32: 4,
  f16: 2,
  bf16: 2,
  i32: 4,
  u32: 4,
  i16: 2,
  u16: 2,
  i8: 1,
  u8: 1,
};

// =============================================================================
// Combined Kernel Thresholds
// =============================================================================

export const DEFAULT_KERNEL_THRESHOLDS = {
  matmul: DEFAULT_MATMUL_THRESHOLDS,
  rmsnorm: DEFAULT_RMSNORM_THRESHOLDS,
  rope: DEFAULT_ROPE_DEFAULTS,
  attention: DEFAULT_ATTENTION_THRESHOLDS,
  fusedMatmul: DEFAULT_FUSED_MATMUL_THRESHOLDS,
  cast: DEFAULT_CAST_THRESHOLDS,
};

// =============================================================================
// Runtime Access
// =============================================================================

let currentThresholds = { ...DEFAULT_KERNEL_THRESHOLDS };

export function getKernelThresholds() {
  return currentThresholds;
}

export function setKernelThresholds(overrides) {
  currentThresholds = {
    ...currentThresholds,
    ...overrides,
    matmul: { ...currentThresholds.matmul, ...overrides.matmul },
    rmsnorm: { ...currentThresholds.rmsnorm, ...overrides.rmsnorm },
    rope: { ...currentThresholds.rope, ...overrides.rope },
    attention: { ...currentThresholds.attention, ...overrides.attention },
    fusedMatmul: { ...currentThresholds.fusedMatmul, ...overrides.fusedMatmul },
    cast: { ...currentThresholds.cast, ...overrides.cast },
  };
}

export function resetKernelThresholds() {
  currentThresholds = { ...DEFAULT_KERNEL_THRESHOLDS };
}
