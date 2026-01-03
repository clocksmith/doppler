import { runMatmulTransposeA, recordMatmulTransposeA, runMatmulBackwardDx, recordMatmulBackwardDx } from './utils.js';

export async function runMatmulBackward(input, weight, gradOutput, options = {}) {
  const {
    M,
    N,
    K,
    transposeB = false,
    computeGradInput = true,
    computeGradWeight = true,
  } = options;
  if (!M || !N || !K) {
    throw new Error('matmul backward requires M, N, and K');
  }
  if (!computeGradInput && !computeGradWeight) {
    throw new Error('matmul backward requires computeGradInput or computeGradWeight');
  }

  let gradInput = null;
  let gradWeight = null;

  if (computeGradInput) {
    // dX = dY @ W^T (or dY @ W depending on transposeB)
    // Fused WGSL kernel handles transpose via uniform flag.
    gradInput = await runMatmulBackwardDx(
      gradOutput,
      weight,
      M,
      K,
      N,
      { transposeB }
    );
  }

  if (computeGradWeight) {
    // dW = X^T @ dY (if Y = X @ W)
    // dW = dY^T @ X (if Y = X @ W^T and we want dW [N, K])
    if (transposeB) {
      // Y = X @ W^T, W is [N, K]
      // dW = dY^T @ X: dY is [M, N], X is [M, K] => dW is [N, K]
      gradWeight = await runMatmulTransposeA(
        gradOutput,
        input,
        N,
        K,
        M
      );
    } else {
      // Y = X @ W, W is [K, N]
      // dW = X^T @ dY: X is [M, K], dY is [M, N] => dW is [K, N]
      gradWeight = await runMatmulTransposeA(
        input,
        gradOutput,
        K,
        N,
        M
      );
    }
  }

  return { gradInput, gradWeight };
}

export async function recordMatmulBackward(
  recorder,
  input,
  weight,
  gradOutput,
  options = {}
) {
  const {
    M,
    N,
    K,
    transposeB = false,
    computeGradInput = true,
    computeGradWeight = true,
  } = options;
  if (!M || !N || !K) {
    throw new Error('matmul backward requires M, N, and K');
  }
  if (!computeGradInput && !computeGradWeight) {
    throw new Error('matmul backward requires computeGradInput or computeGradWeight');
  }

  let gradInput = null;
  let gradWeight = null;

  if (computeGradInput) {
    gradInput = await recordMatmulBackwardDx(
      recorder,
      gradOutput,
      weight,
      M,
      K,
      N,
      { transposeB }
    );
  }

  if (computeGradWeight) {
    if (transposeB) {
      gradWeight = await recordMatmulTransposeA(
        recorder,
        gradOutput,
        input,
        N,
        K,
        M
      );
    } else {
      gradWeight = await recordMatmulTransposeA(
        recorder,
        input,
        gradOutput,
        K,
        N,
        M
      );
    }
  }

  return { gradInput, gradWeight };
}
