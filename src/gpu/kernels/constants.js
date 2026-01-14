


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

  
  Q4K_BLOCK_SIZE: 32,
  Q4K_SUPER_BLOCK_SIZE: 256,
};


export const QUANTIZATION = {
  
  Q4K_BITS: 4.5,
  
  Q4K_BLOCK_BYTES: 144,

  
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


export function getDtypeSize(dtype) {
  return DTYPE_SIZES[dtype];
}


export function calculateBufferSize(shape, dtype) {
  const elements = shape.reduce((a, b) => a * b, 1);
  return elements * getDtypeSize(dtype);
}


export function alignSize(size, alignment = ALIGNMENT.BUFFER) {
  return Math.ceil(size / alignment) * alignment;
}
