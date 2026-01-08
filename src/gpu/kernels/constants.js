/**
 * Kernel Constants - Shared constants for GPU kernels
 *
 * Centralized constants to eliminate magic numbers and improve
 * maintainability across kernel implementations.
 */

/**
 * Workgroup sizes for different kernel types
 * @type {import('./constants.js').WORKGROUP_SIZES}
 */
export const WORKGROUP_SIZES = {
  /** Default workgroup size for most kernels */
  DEFAULT: 256,

  /** Vec4 workgroup thread count (64 threads × 4 elements = 256 elements) */
  VEC4_THREADS: 64,

  /** Attention kernels (large blocks) */
  ATTENTION_LARGE_BLOCK: 64,

  /** Attention kernels (small blocks) */
  ATTENTION_SMALL_BLOCK: 32,

  /** Subgroup size (typical for modern GPUs) */
  SUBGROUP: 32,

  /** RMSNorm workgroup size */
  RMSNORM: 256,

  /** Softmax workgroup size */
  SOFTMAX: 256,

  /** Matmul tile sizes */
  MATMUL_TILE_M: 16,
  MATMUL_TILE_N: 16,
  MATMUL_TILE_K: 16,

  /** MoE workgroup size */
  MOE: 256,
};

/** Derived: Vec4 elements per workgroup (VEC4_THREADS × 4) */
export const VEC4_ELEMENTS_PER_WG = WORKGROUP_SIZES.VEC4_THREADS * 4;  // 256

/**
 * WebGPU limits (spec-level defaults)
 * @type {import('./constants.js').GPU_LIMITS}
 */
export const GPU_LIMITS = {
  /** Max workgroups per dimension (WebGPU minimum) */
  MAX_WORKGROUPS: 65535,
};

/**
 * Memory thresholds for kernel selection (in bytes)
 * @type {import('./constants.js').MEMORY_THRESHOLDS}
 */
export const MEMORY_THRESHOLDS = {
  /** Large attention tier shared memory requirement */
  ATTENTION_LARGE_SHARED: 12288, // 12KB (2 * 32 * 32 * 4 + 32 * 32 * 4)

  /** Small attention tier shared memory requirement (F32) */
  ATTENTION_SMALL_SHARED_F32: 8192, // 8KB (2 * 32 * 32 * 4)

  /** Small attention tier shared memory requirement (F16) */
  ATTENTION_SMALL_SHARED_F16: 4096, // 4KB (2 * 32 * 32 * 2)

  /** Subgroup attention tier shared memory requirement */
  ATTENTION_SUBGROUP_SHARED: 8192, // 2048 * 4 bytes for scores array

  /** Minimum shared memory for any GPU */
  MIN_SHARED_MEMORY: 16384, // 16KB (WebGPU minimum spec)
};

/**
 * Dimension limits for kernel tier selection
 * @type {import('./constants.js').DIMENSION_LIMITS}
 */
export const DIMENSION_LIMITS = {
  /** Maximum head dimension for large attention tier */
  ATTENTION_LARGE_MAX_HEAD_DIM: 64,

  /** Maximum head dimension for small attention tier */
  ATTENTION_SMALL_MAX_HEAD_DIM: 256,

  /** Maximum head dimension for subgroup attention tier */
  ATTENTION_SUBGROUP_MAX_HEAD_DIM: 256,

  /** Maximum sequence length for practical inference */
  MAX_SEQ_LEN: 32768,

  /** Maximum vocab size for typical models */
  MAX_VOCAB_SIZE: 262144, // Gemma 3

  /** Maximum batch size for prefill */
  MAX_BATCH_SIZE: 128,
};

/**
 * Tile sizes for different operations
 * @type {import('./constants.js').TILE_SIZES}
 */
export const TILE_SIZES = {
  /** Attention tile sizes (large) */
  ATTENTION_LARGE_BLOCK_SIZE: 32,
  ATTENTION_LARGE_HEAD_TILE: 32,

  /** Attention tile sizes (small) */
  ATTENTION_SMALL_BLOCK_SIZE: 32,
  ATTENTION_SMALL_HEAD_TILE: 32,

  /** Matmul tile sizes */
  MATMUL_M: 16,
  MATMUL_N: 16,
  MATMUL_K: 16,

  /** Q4K dequant tile sizes */
  Q4K_BLOCK_SIZE: 32,
  Q4K_SUPER_BLOCK_SIZE: 256,
};

/**
 * Quantization constants
 * @type {import('./constants.js').QUANTIZATION}
 */
export const QUANTIZATION = {
  /** Q4K_M bits per weight */
  Q4K_BITS: 4.5,
  /** Q4K block bytes per 256-element super-block */
  Q4K_BLOCK_BYTES: 144,

  /** Q8_0 bits per weight */
  Q8_BITS: 8.5,

  /** F16 bits per weight */
  F16_BITS: 16,

  /** BF16 bits per weight */
  BF16_BITS: 16,

  /** F32 bits per weight */
  F32_BITS: 32,

  /** MXFP4 bits per weight (including shared exponent) */
  MXFP4_BITS: 4,
};

/**
 * Buffer alignment requirements
 * @type {import('./constants.js').ALIGNMENT}
 */
export const ALIGNMENT = {
  /** WebGPU buffer alignment */
  BUFFER: 256,

  /** Uniform buffer alignment */
  UNIFORM: 256,

  /** Storage buffer alignment */
  STORAGE: 256,

  /** Vertex buffer alignment */
  VERTEX: 4,
};

/**
 * Performance tuning constants
 * @type {import('./constants.js').PERFORMANCE}
 */
export const PERFORMANCE = {
  /** Number of warmup runs for benchmarks */
  WARMUP_RUNS: 5,

  /** Number of timed runs for benchmarks */
  TIMED_RUNS: 20,

  /** Default timeout for operations (ms) */
  DEFAULT_TIMEOUT: 120000,

  /** Max buffer pool size per bucket */
  MAX_POOL_SIZE_PER_BUCKET: 8,

  /** Max total pooled buffers */
  MAX_TOTAL_POOLED_BUFFERS: 64,
};

/**
 * Dtype size mappings (in bytes)
 * @type {import('./constants.js').DTYPE_SIZES}
 */
export const DTYPE_SIZES = {
  u8: 1,
  i8: 1,
  u16: 2,
  i16: 2,
  f16: 2,
  bf16: 2,
  u32: 4,
  i32: 4,
  f32: 4,
  f64: 8,
};

/**
 * Get dtype size in bytes
 * @param {import('./constants.js').DType} dtype
 * @returns {number}
 */
export function getDtypeSize(dtype) {
  return DTYPE_SIZES[dtype];
}

/**
 * Calculate buffer size for tensor
 * @param {number[]} shape
 * @param {import('./constants.js').DType} dtype
 * @returns {number}
 */
export function calculateBufferSize(shape, dtype) {
  const elements = shape.reduce((a, b) => a * b, 1);
  return elements * getDtypeSize(dtype);
}

/**
 * Round size up to alignment boundary
 * @param {number} size
 * @param {number} [alignment]
 * @returns {number}
 */
export function alignSize(size, alignment = ALIGNMENT.BUFFER) {
  return Math.ceil(size / alignment) * alignment;
}
