/**
 * Kernel Constants - Shared constants for GPU kernels
 *
 * Centralized constants to eliminate magic numbers and improve
 * maintainability across kernel implementations.
 */

/**
 * Workgroup sizes for different kernel types
 */
export declare const WORKGROUP_SIZES: {
  /** Default workgroup size for most kernels */
  readonly DEFAULT: 256;

  /** Vec4 workgroup thread count (64 threads × 4 elements = 256 elements) */
  readonly VEC4_THREADS: 64;

  /** Attention kernels (large blocks) */
  readonly ATTENTION_LARGE_BLOCK: 32;

  /** Attention kernels (small blocks) */
  readonly ATTENTION_SMALL_BLOCK: 32;

  /** Subgroup size (typical for modern GPUs) */
  readonly SUBGROUP: 32;

  /** RMSNorm workgroup size */
  readonly RMSNORM: 256;

  /** Softmax workgroup size */
  readonly SOFTMAX: 256;

  /** Matmul tile sizes */
  readonly MATMUL_TILE_M: 16;
  readonly MATMUL_TILE_N: 16;
  readonly MATMUL_TILE_K: 16;

  /** MoE workgroup size */
  readonly MOE: 256;
};

/** Derived: Vec4 elements per workgroup (VEC4_THREADS × 4) */
export declare const VEC4_ELEMENTS_PER_WG: number;

/**
 * WebGPU limits (spec-level defaults)
 */
export declare const GPU_LIMITS: {
  /** Max workgroups per dimension (WebGPU minimum) */
  readonly MAX_WORKGROUPS: 65535;
};

/**
 * Tile sizes for different operations
 */
export declare const TILE_SIZES: {
  /** Attention tile sizes (large) */
  readonly ATTENTION_LARGE_BLOCK_SIZE: 32;
  readonly ATTENTION_LARGE_HEAD_TILE: 64;

  /** Attention tile sizes (small) */
  readonly ATTENTION_SMALL_BLOCK_SIZE: 32;
  readonly ATTENTION_SMALL_HEAD_TILE: 32;

  /** Matmul tile sizes */
  readonly MATMUL_M: 16;
  readonly MATMUL_N: 16;
  readonly MATMUL_K: 16;

  /** Q4K dequant tile sizes */
  readonly Q4K_BLOCK_SIZE: 32;
  readonly Q4K_SUPER_BLOCK_SIZE: 256;
};

/**
 * Quantization constants
 */
export declare const QUANTIZATION: {
  /** Q4K_M bits per weight */
  readonly Q4K_BITS: 4.5;
  /** Q4K block bytes per 256-element super-block */
  readonly Q4K_BLOCK_BYTES: 144;

  /** Q8_0 bits per weight */
  readonly Q8_BITS: 8.5;

  /** F16 bits per weight */
  readonly F16_BITS: 16;

  /** BF16 bits per weight */
  readonly BF16_BITS: 16;

  /** F32 bits per weight */
  readonly F32_BITS: 32;

  /** MXFP4 bits per weight (including shared exponent) */
  readonly MXFP4_BITS: 4;
};

/**
 * Buffer alignment requirements
 */
export declare const ALIGNMENT: {
  /** WebGPU buffer alignment */
  readonly BUFFER: 256;

  /** Uniform buffer alignment */
  readonly UNIFORM: 256;

  /** Storage buffer alignment */
  readonly STORAGE: 256;

  /** Vertex buffer alignment */
  readonly VERTEX: 4;
};

/**
 * Performance tuning constants
 */
export declare const PERFORMANCE: {
  /** Number of warmup runs for benchmarks */
  readonly WARMUP_RUNS: 5;

  /** Number of timed runs for benchmarks */
  readonly TIMED_RUNS: 20;

  /** Default timeout for operations (ms) */
  readonly DEFAULT_TIMEOUT: 120000;

  /** Max buffer pool size per bucket */
  readonly MAX_POOL_SIZE_PER_BUCKET: 8;

  /** Max total pooled buffers */
  readonly MAX_TOTAL_POOLED_BUFFERS: 64;
};

/**
 * Dtype size mappings (in bytes)
 */
export declare const DTYPE_SIZES: {
  readonly u8: 1;
  readonly i8: 1;
  readonly u16: 2;
  readonly i16: 2;
  readonly f16: 2;
  readonly bf16: 2;
  readonly u32: 4;
  readonly i32: 4;
  readonly f32: 4;
  readonly f64: 8;
};

export type DType = keyof typeof DTYPE_SIZES;

/**
 * Get dtype size in bytes
 */
export declare function getDtypeSize(dtype: DType): number;

/**
 * Calculate buffer size for tensor
 */
export declare function calculateBufferSize(shape: number[], dtype: DType): number;

/**
 * Round size up to alignment boundary
 */
export declare function alignSize(size: number, alignment?: number): number;
