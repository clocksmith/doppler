/**
 * Kernel Thresholds Schema
 *
 * Centralized configuration for kernel selection thresholds and magic numbers.
 * These values control when variant selection switches between kernel implementations.
 *
 * @module config/schema/kernel-thresholds
 */

// =============================================================================
// Matmul Thresholds
// =============================================================================

/**
 * Thresholds for matrix multiplication kernel variant selection.
 */
export interface MatmulThresholdsSchema {
  /**
   * N dimension threshold for selecting multicol GEMV variants.
   * When N >= threshold, use multicol variant for reduced workgroup count.
   * @default 256
   */
  multicolThreshold: number;
}

export const DEFAULT_MATMUL_THRESHOLDS: MatmulThresholdsSchema = {
  multicolThreshold: 256,
};

// =============================================================================
// RMSNorm Thresholds
// =============================================================================

/**
 * Thresholds for RMSNorm kernel variant selection.
 */
export interface RmsnormThresholdsSchema {
  /**
   * Hidden size threshold for selecting small vs default variant.
   * When hiddenSize <= threshold, use small variant (single workgroup).
   * @default 256
   */
  smallThreshold: number;
}

export const DEFAULT_RMSNORM_THRESHOLDS: RmsnormThresholdsSchema = {
  smallThreshold: 256,
};

// =============================================================================
// RoPE Thresholds
// =============================================================================

/**
 * Default values for RoPE (Rotary Position Embedding) kernel.
 */
export interface RopeDefaultsSchema {
  /**
   * Default theta value for RoPE frequency computation.
   * Most models use 10000.0; some (Gemma 3) use higher values.
   * @default 10000.0
   */
  defaultTheta: number;

  /**
   * Default uniform buffer size in bytes for RoPE params.
   * @default 32
   */
  uniformSize: number;
}

export const DEFAULT_ROPE_DEFAULTS: RopeDefaultsSchema = {
  defaultTheta: 10000.0,
  uniformSize: 32,
};

// =============================================================================
// Attention Thresholds
// =============================================================================

/**
 * Thresholds for attention kernel variant selection.
 */
export interface AttentionThresholdsSchema {
  /**
   * Maximum KV length before switching from chunked to streaming attention.
   * Used by decode_chunked_f16kv variant.
   * @default 2048
   */
  chunkedMaxKVLen: number;

  /**
   * Minimum head dimension for chunked attention kernel eligibility.
   * Chunked kernels require headDim >= this value.
   * @default 128
   */
  minHeadDimForChunked: number;

  /**
   * Head dimension thresholds for tier selection.
   * tier3: headDim <= 64, tier2: headDim <= 128, tier1: headDim <= 256
   */
  tierHeadDimLimits: {
    tier3: number;
    tier2: number;
    tier1: number;
  };

  /**
   * Minimum shared memory requirements per tier (in bytes).
   */
  tierMinSharedMemory: {
    tier3: number;
    tier2: number;
    tier1: number;
  };
}

export const DEFAULT_ATTENTION_THRESHOLDS: AttentionThresholdsSchema = {
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

/**
 * Thresholds for fused matmul+norm kernel variant selection.
 */
export interface FusedMatmulThresholdsSchema {
  /**
   * Maximum N dimension for "medium" (multi-column) fused variant.
   * Beyond this, fall back to separate kernels or alternative dispatch.
   * @default 4096
   */
  maxMediumN: number;

  /**
   * Columns per workgroup for multi-column dispatch.
   * Each workgroup processes this many output columns.
   * @default 4
   */
  colsPerWg: number;
}

export const DEFAULT_FUSED_MATMUL_THRESHOLDS: FusedMatmulThresholdsSchema = {
  maxMediumN: 4096,
  colsPerWg: 4,
};

// =============================================================================
// Cast Thresholds
// =============================================================================

/**
 * Configuration for cast kernel dispatch.
 */
export interface CastThresholdsSchema {
  /**
   * Maximum workgroups per dimension before falling back to 2D dispatch.
   * @default 65535
   */
  maxWorkgroupsPerDim: number;
}

export const DEFAULT_CAST_THRESHOLDS: CastThresholdsSchema = {
  maxWorkgroupsPerDim: 65535,
};

// =============================================================================
// Dtype Size Constants
// =============================================================================

/**
 * Bytes per element for each data type.
 */
export const DTYPE_SIZES: Record<string, number> = {
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

/**
 * All kernel thresholds in a single configuration object.
 */
export interface KernelThresholdsConfigSchema {
  matmul: MatmulThresholdsSchema;
  rmsnorm: RmsnormThresholdsSchema;
  rope: RopeDefaultsSchema;
  attention: AttentionThresholdsSchema;
  fusedMatmul: FusedMatmulThresholdsSchema;
  cast: CastThresholdsSchema;
}

export const DEFAULT_KERNEL_THRESHOLDS: KernelThresholdsConfigSchema = {
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

/**
 * Current kernel thresholds configuration.
 * Can be overridden at runtime for testing or optimization.
 */
let currentThresholds: KernelThresholdsConfigSchema = { ...DEFAULT_KERNEL_THRESHOLDS };

/**
 * Get the current kernel thresholds configuration.
 */
export function getKernelThresholds(): KernelThresholdsConfigSchema {
  return currentThresholds;
}

/**
 * Override kernel thresholds (merges with current config).
 */
export function setKernelThresholds(overrides: Partial<KernelThresholdsConfigSchema>): void {
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

/**
 * Reset kernel thresholds to defaults.
 */
export function resetKernelThresholds(): void {
  currentThresholds = { ...DEFAULT_KERNEL_THRESHOLDS };
}
