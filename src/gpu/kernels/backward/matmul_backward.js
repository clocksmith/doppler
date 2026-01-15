import { runMatmul } from '../matmul.js';
import { runTranspose } from '../transpose.js';

export async function runMatmulBackward(input, weight, gradOutput, options = {}) {
  const { M, N, K, transposeB = false } = options;
  if (!M || !N || !K) {
    throw new Error('matmul backward requires M, N, and K');
  }

  const weightTensor = weight;
  const weightTransposed = await runTranspose(weightTensor, K, N);
  const gradInput = await runMatmul(
    gradOutput,
    weightTransposed,
    M,
    K,
    N,
    { transposeB: false }
  );

  const inputTransposed = await runTranspose(input, M, K);
  const gradWeight = await runMatmul(
    inputTransposed,
    gradOutput,
    K,
    N,
    M,
    { transposeB }
  );

  return { gradInput, gradWeight };
}

export async function recordMatmulBackward() {
  throw new Error('recordMatmulBackward not implemented for multi-output matmul backward');
}
