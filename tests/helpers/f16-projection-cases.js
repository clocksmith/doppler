import { float32ToFloat16 } from '../kernels/reference/dequant.js';

// Exact binary input/weight fractions isolate accumulation and layout errors.
export function createF16ProjectionCases({ transpose: requiredTranspose = null } = {}) {
  const shapes = [
    { M: 1, N: 1, K: 1, tileSize: 1, transpose: 0, alpha: 1 },
    { M: 3, N: 5, K: 7, tileSize: 3, transpose: 0, alpha: -0.5 },
    { M: 17, N: 19, K: 33, tileSize: 16, transpose: 1, alpha: 2 },
    { M: 9, N: 7, K: 129, tileSize: 8, transpose: 0, alpha: 1 },
    { M: 1, N: 16, K: 4096, tileSize: 16, transpose: 1, alpha: 1 },
    { M: 3, N: 5, K: 1024, tileSize: 16, transpose: 1, alpha: 1, magnitude: 2 ** 70 },
    { M: 2, N: 3, K: 32, tileSize: 8, transpose: 0, alpha: 1, magnitude: 2 ** -60 },
    { M: 65, N: 67, K: 33, tileSize: 16, transpose: 1, alpha: -2 },
  ];
  const encode = array => Buffer.from(array.buffer).toString('base64');
  return shapes.map((shape, index) => {
    if (requiredTranspose !== null) shape = { ...shape, transpose: requiredTranspose };
    const { M, N, K, transpose, alpha, magnitude = 1 } = shape;
    const input = Float32Array.from({ length: M * K }, (_, i) => ((i * 7 + 3) % 31 - 15) / 16 * magnitude);
    const weights = Float32Array.from({ length: N * K }, (_, i) => ((i * 11 + 5) % 17 - 8) / 16);
    const packed = new Uint16Array(N * K);
    for (let n = 0; n < N; n++) for (let k = 0; k < K; k++) {
      packed[transpose ? n * K + k : k * N + n] = float32ToFloat16(weights[n * K + k]);
    }
    const expected = new Float64Array(M * N), bounds = new Float64Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let sum = 0, magnitudeSum = 0;
      for (let k = 0; k < K; k++) {
        const term = input[m * K + k] * weights[n * K + k];
        sum += term; magnitudeSum += Math.abs(term);
      }
      expected[m * N + n] = sum * alpha;
      bounds[m * N + n] = Math.max(Number.MIN_VALUE, Math.abs(alpha) * magnitudeSum * 2e-6);
    }
    return { ...shape, name: `synthetic-${index}`, kind: 'synthetic', inputF32: encode(input),
      weightF16: encode(packed), expectedF64: encode(expected), errorBoundF64: encode(bounds) };
  });
}
