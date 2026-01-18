/**
 * Dtype Utilities
 *
 * Data type conversion utilities for tensor loading.
 *
 * @module loader/dtype-utils
 */

import type { TensorLocation } from './loader-types.js';

/**
 * Convert F16 (half precision) to F32 (single precision)
 */
export declare function f16ToF32(h: number): number;

/**
 * Convert BF16 buffer to F32 on GPU
 */
export declare function convertBF16ToF32GPU(
  srcBuffer: GPUBuffer,
  numElements: number,
  name: string
): Promise<GPUBuffer>;

/**
 * Decide whether a quantized tensor should be dequantized directly to f16.
 * Returns true for matmul weights (projections, FFN, lm_head, embeddings).
 */
export declare function shouldDequantizeToF16(location: TensorLocation): boolean;

/**
 * Apply layout metadata to a GPU buffer if the tensor has column-major storage.
 * Note: Layout is now tracked via WeightBuffer for matmul weights.
 * This function is kept for API compatibility but is a no-op for non-matmul weights (norms).
 */
export declare function applyBufferLayout(buffer: GPUBuffer, _location: TensorLocation): GPUBuffer;
