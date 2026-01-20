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
// Softmax Thresholds
// =============================================================================

export const DEFAULT_SOFTMAX_THRESHOLDS = {
  smallThreshold: 256,
};

// =============================================================================
// FFN Thresholds
// =============================================================================

export const DEFAULT_FFN_THRESHOLDS = {
  multiOutputThreshold: 1024,
};

// =============================================================================
// Sample Thresholds
// =============================================================================

export const DEFAULT_SAMPLE_THRESHOLDS = {
  argmaxReduceVocabThreshold: 65536,
  singlePassTopKThreshold: 100,
};

// =============================================================================
// RoPE Thresholds
// =============================================================================

export const DEFAULT_ROPE_DEFAULTS = {
  defaultTheta: 10000.0,
  uniformSize: 32,
  defaultStartPos: 0,
};

// =============================================================================
// Kernel Tuning Defaults
// =============================================================================

export const DEFAULT_TUNER_LIMITS = {
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeInvocationsPerWorkgroup: 256,
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
  largeMaxHeadDim: 64,
  smallMaxHeadDim: 256,
  subgroupMaxHeadDim: 256,
  largeSharedF32: 20480,
  largeSharedF16: 49152,
  smallSharedF32: 8192,
  smallSharedF16: 4096,
  subgroupShared: 8192,
};

// =============================================================================
// Fused Matmul Thresholds
// =============================================================================

export const DEFAULT_FUSED_MATMUL_THRESHOLDS = {
  maxMediumN: 4096,
  maxMediumK: 8192,
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
  float32: 4,
  f16: 2,
  float16: 2,
  bf16: 2,
  bfloat16: 2,
  i32: 4,
  int32: 4,
  u32: 4,
  uint32: 4,
  i16: 2,
  int16: 2,
  u16: 2,
  uint16: 2,
  i8: 1,
  int8: 1,
  u8: 1,
  uint8: 1,
};

export function getDtypeSize(dtype) {
  const size = DTYPE_SIZES[dtype?.toLowerCase()];
  if (size === undefined) {
    throw new Error(`Unknown dtype: "${dtype}". Valid: ${Object.keys(DTYPE_SIZES).join(', ')}`);
  }
  return size;
}

// =============================================================================
// Combined Kernel Thresholds
// =============================================================================

export const DEFAULT_KERNEL_THRESHOLDS = {
  matmul: DEFAULT_MATMUL_THRESHOLDS,
  rmsnorm: DEFAULT_RMSNORM_THRESHOLDS,
  softmax: DEFAULT_SOFTMAX_THRESHOLDS,
  ffn: DEFAULT_FFN_THRESHOLDS,
  sample: DEFAULT_SAMPLE_THRESHOLDS,
  rope: DEFAULT_ROPE_DEFAULTS,
  attention: DEFAULT_ATTENTION_THRESHOLDS,
  fusedMatmul: DEFAULT_FUSED_MATMUL_THRESHOLDS,
  cast: DEFAULT_CAST_THRESHOLDS,
  tuner: DEFAULT_TUNER_LIMITS,
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
    softmax: { ...currentThresholds.softmax, ...overrides.softmax },
    ffn: { ...currentThresholds.ffn, ...overrides.ffn },
    sample: { ...currentThresholds.sample, ...overrides.sample },
    rope: { ...currentThresholds.rope, ...overrides.rope },
    attention: { ...currentThresholds.attention, ...overrides.attention },
    fusedMatmul: { ...currentThresholds.fusedMatmul, ...overrides.fusedMatmul },
    cast: { ...currentThresholds.cast, ...overrides.cast },
    tuner: { ...currentThresholds.tuner, ...overrides.tuner },
  };
}

export function resetKernelThresholds() {
  currentThresholds = { ...DEFAULT_KERNEL_THRESHOLDS };
}
