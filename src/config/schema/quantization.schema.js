
// Q4K super-block size (invariant - format spec constant)
// WGSL kernels also define this as `const QK_K: u32 = 256u;`
// JS and WGSL must agree on this value.
export const QK_K = 256;

// Q4K block size in bytes (144 bytes per 256-element super-block)
export const Q4K_BLOCK_BYTES = 144;

// Q6K block size in bytes (210 bytes per 256-element super-block)
export const Q6K_BLOCK_BYTES = 210;

// Q8_0 block size in bytes (34 bytes per 32-element block)
export const Q8_0_BLOCK_BYTES = 34;

// Q8_0 block size in elements
export const Q8_0_BLOCK_SIZE = 32;

// K_SCALE_SIZE for Q4K (12 bytes for scales/mins metadata)
export const K_SCALE_SIZE = 12;

// Q4K block size in bytes (same as Q4K_BLOCK_BYTES, exported for compatibility)
export const QK4_K_BLOCK_SIZE = Q4K_BLOCK_BYTES;

// Pad size to Q4K super-block alignment
export function padToQ4KBlock(size) {
  return Math.ceil(size / QK_K) * QK_K;
}

// Calculate number of Q4K blocks for a given number of elements
export function q4kBlockCount(numElements) {
  return Math.ceil(numElements / QK_K);
}
