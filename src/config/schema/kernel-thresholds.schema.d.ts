/**
 * Kernel Thresholds Schema
 *
 * Centralized configuration for kernel selection thresholds and magic numbers.
 * These values control when variant selection switches between kernel implementations.
 *
 * @module config/schema/kernel-thresholds
 */

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

export declare const DEFAULT_MATMUL_THRESHOLDS: MatmulThresholdsSchema;

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

export declare const DEFAULT_RMSNORM_THRESHOLDS: RmsnormThresholdsSchema;

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

export declare const DEFAULT_ROPE_DEFAULTS: RopeDefaultsSchema;

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

export declare const DEFAULT_ATTENTION_THRESHOLDS: AttentionThresholdsSchema;

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

export declare const DEFAULT_FUSED_MATMUL_THRESHOLDS: FusedMatmulThresholdsSchema;

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

export declare const DEFAULT_CAST_THRESHOLDS: CastThresholdsSchema;

/**
 * Bytes per element for each data type.
 */
export declare const DTYPE_SIZES: Record<string, number>;

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

export declare const DEFAULT_KERNEL_THRESHOLDS: KernelThresholdsConfigSchema;

/**
 * Get the current kernel thresholds configuration.
 */
export declare function getKernelThresholds(): KernelThresholdsConfigSchema;

/**
 * Override kernel thresholds (merges with current config).
 */
export declare function setKernelThresholds(overrides: Partial<KernelThresholdsConfigSchema>): void;

/**
 * Reset kernel thresholds to defaults.
 */
export declare function resetKernelThresholds(): void;
