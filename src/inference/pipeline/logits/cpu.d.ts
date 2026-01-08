/**
 * CPU reference implementations for logits computation.
 *
 * Provides CPU fallback implementations for RMSNorm, matmul, and softcapping.
 * These are used when GPU is unavailable or for validation.
 *
 * @module inference/pipeline/logits/cpu
 */

/**
 * CPU RMSNorm implementation.
 *
 * Computes: output[i] = (x[i] / rms) * weight[i]
 * where rms = sqrt(mean(x^2) + eps)
 *
 * @param x - Input tensor
 * @param weight - Norm weights
 * @param eps - Epsilon for numerical stability
 * @returns Normalized tensor
 */
export function rmsNormCPU(
  x: Float32Array,
  weight: Float32Array,
  eps?: number
): Float32Array;

/**
 * Convert a single float16 value to float32.
 *
 * @param h - Float16 value as uint16
 * @returns Float32 value
 */
export function f16ToF32(h: number): number;

/**
 * Convert a buffer of float16 values to float32.
 *
 * @param data - ArrayBuffer containing float16 data
 * @returns Float32Array with converted values
 */
export function f16BufferToF32(data: ArrayBuffer): Float32Array;

/**
 * CPU matmul implementation (fallback for non-GPU).
 *
 * Computes: output = input @ weight^T
 * Input: [M, K], Weight: [N, K] (row) or [K, N] (column), Output: [M, N]
 *
 * @param input - Input tensor [M, K]
 * @param weight - Weight tensor [N, K] (row) or [K, N] (column)
 * @param M - Batch size (num tokens)
 * @param N - Output size (vocab size)
 * @param K - Hidden size
 * @param layout - Weight layout ('row' or 'column')
 * @param weightStride - Optional stride override for weight indexing
 * @returns Output tensor [M, N]
 */
export function matmulCPU(
  input: Float32Array,
  weight: Float32Array,
  M: number,
  N: number,
  K: number,
  layout?: 'row' | 'column',
  weightStride?: number | null
): Float32Array;

/**
 * Apply softcapping to logits (Gemma 2 style).
 *
 * Computes: logits = tanh(logits / cap) * cap
 *
 * This bounds logits to [-cap, cap] with smooth transitions,
 * preventing extreme values from dominating softmax.
 *
 * @param logits - Logits tensor to modify in-place
 * @param cap - Softcap value (Gemma 2 default: 30.0)
 */
export function applySoftcapping(logits: Float32Array, cap: number): void;
