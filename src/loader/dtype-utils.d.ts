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
export declare function shouldDequantizeToF16(name: string): boolean;

/**
 * Check if a weight is an embedding weight (needs column layout for LM head matmul).
 * GGUF stores all weights as [N,K] (transposed). For layer weights, we need transposeB=true
 * to compute A@W = A@W.T^T. But for embeddings used as LM head, we need transposeB=false
 * to compute hidden@E.T directly (the embedding IS already transposed in GGUF).
 */
export declare function isEmbeddingWeight(name: string): boolean;

/**
 * Apply layout metadata to a GPU buffer if the tensor has column-major storage.
 * Note: Layout is now tracked via WeightBuffer for matmul weights.
 * This function is kept for API compatibility but is a no-op for non-matmul weights (norms).
 */
export declare function applyBufferLayout(buffer: GPUBuffer, _location: TensorLocation): GPUBuffer;

/**
 * Apply +1 offset to norm weights for Gemma models.
 *
 * IMPORTANT: actualNumElements must be provided to avoid reading garbage padding
 * from the buffer pool's power-of-2 bucketing.
 *
 * @param bufferDtype - Optional dtype for GPU buffer (defaults to 'f32')
 */
export declare function applyNormWeightOffset(
  tensor: GPUBuffer | Float32Array,
  actualNumElements?: number,
  normOffsetDebugLogged?: boolean,
  bufferDtype?: 'f16' | 'f32' | 'bf16'
): Promise<{ tensor: GPUBuffer | Float32Array; debugLogged: boolean }>;

/**
 * Find alternative tensor name (handles different naming conventions).
 * Returns null if no alternative is found.
 */
export declare function findAlternativeTensorName(
  name: string,
  tensorLocations: Map<string, TensorLocation>
): string | null;
