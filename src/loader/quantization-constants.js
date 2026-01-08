/**
 * Centralized Quantization Format Constants
 *
 * Single source of truth for quantization block sizes and formats.
 * Import these instead of redefining locally.
 *
 * @module loader/quantization-constants
 */

// Re-export Q4K constants from quantizer (source of truth)
export { QK_K, QK4_K_BLOCK_SIZE, K_SCALE_SIZE } from '../converter/quantizer.js';

/** Q4K block size in bytes (alias for clarity) */
export const Q4K_BLOCK_BYTES = 144;

/** Q6K block size in bytes (210 bytes per 256 weights) */
export const Q6K_BLOCK_BYTES = 210;

/** Q8_0 block size in bytes (34 bytes per 32 weights: 2 byte scale + 32 byte qs) */
export const Q8_0_BLOCK_BYTES = 34;

/** Q8_0 block size in elements (32 weights per block) */
export const Q8_0_BLOCK_SIZE = 32;
