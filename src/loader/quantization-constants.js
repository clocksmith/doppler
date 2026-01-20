

// Re-export quantization constants from schema (single source of truth)
export {
  QK_K,
  QK4_K_BLOCK_SIZE,
  K_SCALE_SIZE,
  Q4K_BLOCK_BYTES,
  Q6K_BLOCK_BYTES,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_SIZE,
  padToQ4KBlock,
  q4kBlockCount,
} from '../config/schema/index.js';
