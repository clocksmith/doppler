


import { QK_K, Q4K_BLOCK_BYTES } from '../../config/schema/index.js';

export const WORKGROUP_SIZES = {

  DEFAULT: 256,

  
  VEC4_THREADS: 64,

  
  ATTENTION_LARGE_BLOCK: 32,

  
  ATTENTION_SMALL_BLOCK: 32,

  
  SUBGROUP: 32,

  
  RMSNORM: 256,

  
  SOFTMAX: 256,

  
  MATMUL_TILE_M: 16,
  MATMUL_TILE_N: 16,
  MATMUL_TILE_K: 16,

  
  MOE: 256,
};


export const VEC4_ELEMENTS_PER_WG = WORKGROUP_SIZES.VEC4_THREADS * 4;  // 256


export const GPU_LIMITS = {
  
  MAX_WORKGROUPS: 65535,
};

export const TILE_SIZES = {

  ATTENTION_LARGE_BLOCK_SIZE: 32,
  ATTENTION_LARGE_HEAD_TILE: 64,


  ATTENTION_SMALL_BLOCK_SIZE: 32,
  ATTENTION_SMALL_HEAD_TILE: 32,


  MATMUL_M: 16,
  MATMUL_N: 16,
  MATMUL_K: 16,

  // Q4K tile size (sub-blocks within super-block)
  Q4K_BLOCK_SIZE: 32,
  // Q4K super-block size imported from schema (single source of truth)
  Q4K_SUPER_BLOCK_SIZE: QK_K,
};


export const QUANTIZATION = {

  Q4K_BITS: 4.5,
  // Q4K block bytes imported from schema (single source of truth)
  Q4K_BLOCK_BYTES,

  
  Q8_BITS: 8.5,

  
  F16_BITS: 16,

  
  BF16_BITS: 16,

  
  F32_BITS: 32,

  
  MXFP4_BITS: 4,
};


export const ALIGNMENT = {
  
  BUFFER: 256,

  
  UNIFORM: 256,

  
  STORAGE: 256,

  
  VERTEX: 4,
};


export const PERFORMANCE = {
  
  WARMUP_RUNS: 5,

  
  TIMED_RUNS: 20,

  
  DEFAULT_TIMEOUT: 120000,

  
  MAX_POOL_SIZE_PER_BUCKET: 8,

  
  MAX_TOTAL_POOLED_BUFFERS: 64,
};


// DTYPE_SIZES and getDtypeSize moved to config/schema/kernel-thresholds.schema.js
// Import from config/schema/index.js for the canonical source


export function alignSize(size, alignment = ALIGNMENT.BUFFER) {
  return Math.ceil(size / alignment) * alignment;
}
