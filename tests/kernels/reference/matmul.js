


export function matmulRef(A, B, M, N, K, alpha = 1.0) {
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


export function batchMatmulRef(A, B, batch, M, N, K) {
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


export function matvecRef(A, x, M, K) {
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
