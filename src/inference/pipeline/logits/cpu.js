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
 * @param {Float32Array} x - Input tensor
 * @param {Float32Array} weight - Norm weights
 * @param {number} [eps=1e-5] - Epsilon for numerical stability
 * @returns {Float32Array} Normalized tensor
 */
export function rmsNormCPU(
  x,
  weight,
  eps = 1e-5
) {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += x[i] * x[i];
  }
  const rms = Math.sqrt(sumSq / n + eps);

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = (x[i] / rms) * weight[i % weight.length];
  }
  return result;
}

/**
 * Convert a single float16 value to float32.
 *
 * @param {number} h - Float16 value as uint16
 * @returns {number} Float32 value
 */
export function f16ToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

/**
 * Convert a buffer of float16 values to float32.
 *
 * @param {ArrayBuffer} data - ArrayBuffer containing float16 data
 * @returns {Float32Array} Float32Array with converted values
 */
export function f16BufferToF32(data) {
  const u16 = new Uint16Array(data);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    out[i] = f16ToF32(u16[i]);
  }
  return out;
}

/**
 * CPU matmul implementation (fallback for non-GPU).
 *
 * Computes: output = input @ weight^T
 * Input: [M, K], Weight: [N, K] (row) or [K, N] (column), Output: [M, N]
 *
 * @param {Float32Array} input - Input tensor [M, K]
 * @param {Float32Array} weight - Weight tensor [N, K] (row) or [K, N] (column)
 * @param {number} M - Batch size (num tokens)
 * @param {number} N - Output size (vocab size)
 * @param {number} K - Hidden size
 * @param {'row' | 'column'} [layout='row'] - Weight layout ('row' or 'column')
 * @param {number | null} [weightStride] - Optional stride override for weight indexing
 * @returns {Float32Array} Output tensor [M, N]
 */
export function matmulCPU(
  input,
  weight,
  M,
  N,
  K,
  layout = 'row',
  weightStride
) {
  const result = new Float32Array(M * N);
  const stride = weightStride ?? (layout === 'row' ? K : N);

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        // Row layout: weight is [N, K] (vocab x hidden).
        // Column layout: weight is [K, N] (hidden x vocab).
        const weightIndex = layout === 'row'
          ? n * stride + k
          : k * stride + n;
        sum += input[m * K + k] * weight[weightIndex];
      }
      result[m * N + n] = sum;
    }
  }
  return result;
}

/**
 * Apply softcapping to logits (Gemma 2 style).
 *
 * Computes: logits = tanh(logits / cap) * cap
 *
 * This bounds logits to [-cap, cap] with smooth transitions,
 * preventing extreme values from dominating softmax.
 *
 * @param {Float32Array} logits - Logits tensor to modify in-place
 * @param {number} cap - Softcap value (Gemma 2 default: 30.0)
 * @returns {void}
 */
export function applySoftcapping(logits, cap) {
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.tanh(logits[i] / cap) * cap;
  }
}
