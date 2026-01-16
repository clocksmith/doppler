


export function rmsNormCPU(
  x,
  weight,
  eps = 1e-5,
  rmsNormWeightOffset = false
) {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += x[i] * x[i];
  }
  const rms = Math.sqrt(sumSq / n + eps);

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const w = weight[i % weight.length];
    const scale = rmsNormWeightOffset ? 1 + w : w;
    result[i] = (x[i] / rms) * scale;
  }
  return result;
}


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


export function f16BufferToF32(data) {
  const u16 = new Uint16Array(data);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    out[i] = f16ToF32(u16[i]);
  }
  return out;
}


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


export function applySoftcapping(logits, cap) {
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.tanh(logits[i] / cap) * cap;
  }
}
