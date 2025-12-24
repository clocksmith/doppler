/**
 * Reference Matrix Multiplication
 */

/**
 * Reference matrix multiplication
 * C = alpha * A @ B
 *
 * @param A Matrix A [M x K] row-major
 * @param B Matrix B [K x N] row-major
 * @param M Rows of A
 * @param N Cols of B
 * @param K Shared dimension
 * @param alpha Scaling factor
 * @returns Result C [M x N]
 */
export function matmulRef(
  A: Float32Array,
  B: Float32Array,
  M: number,
  N: number,
  K: number,
  alpha: number = 1.0
): Float32Array {
  const C = new Float32Array(M * N);

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum * alpha;
    }
  }

  return C;
}

/**
 * Batched matrix multiplication
 * @param A [batch, M, K]
 * @param B [batch, K, N]
 * @param batch Batch size
 * @param M Rows of each A
 * @param N Cols of each B
 * @param K Shared dimension
 * @returns [batch, M, N]
 */
export function batchMatmulRef(
  A: Float32Array,
  B: Float32Array,
  batch: number,
  M: number,
  N: number,
  K: number
): Float32Array {
  const C = new Float32Array(batch * M * N);
  const strideA = M * K;
  const strideB = K * N;
  const strideC = M * N;

  for (let b = 0; b < batch; b++) {
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += A[b * strideA + m * K + k] * B[b * strideB + k * N + n];
        }
        C[b * strideC + m * N + n] = sum;
      }
    }
  }

  return C;
}

/**
 * Matrix-vector multiplication
 * y = A @ x
 * @param A [M, K]
 * @param x [K]
 * @param M Number of rows
 * @param K Number of columns
 * @returns [M]
 */
export function matvecRef(A: Float32Array, x: Float32Array, M: number, K: number): Float32Array {
  const y = new Float32Array(M);

  for (let m = 0; m < M; m++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      sum += A[m * K + k] * x[k];
    }
    y[m] = sum;
  }

  return y;
}

export default matmulRef;
