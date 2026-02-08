import { runMatmul, recordMatmul } from '../matmul.js';
import { runTranspose, recordTranspose } from '../transpose.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { runMatmulTransposeA, recordMatmulTransposeA } from './utils.js';

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
    // dX = dY @ W (if Y = X @ W^T)
    // dX = dY @ W^T (if Y = X @ W)
    if (transposeB) {
      // Y = X @ W^T => dX = dY @ W
      // dY is [M, N], W is [N, K] => dX is [M, K]
      gradInput = await runMatmul(
        gradOutput,
        weight,
        M,
        K,
        N,
        { transposeB: false }
      );
    } else {
      // Y = X @ W => dX = dY @ W^T
      // dY is [M, N], W is [K, N] => dX is [M, K]
      // Using runMatmul with transposeB=true
      gradInput = await runMatmul(
        gradOutput,
        weight,
        M,
        K,
        N,
        { transposeB: true }
      );
    }
  }

  if (computeGradWeight) {
    // dW = X^T @ dY (if Y = X @ W)
    // dW = dY^T @ X (if Y = X @ W^T and we want dW [N, K])
    if (transposeB) {
      // Y = X @ W^T
      // dW^T = X^T @ dY => dW = dY^T @ X
      // dY is [M, N], X is [M, K] => dW is [N, K]
      // Use specialized transposeA matmul: dW = dY^T @ X
      gradWeight = await runMatmulTransposeA(
        gradOutput,
        input,
        N,
        K,
        M
      );
    } else {
      // Y = X @ W
      // dW = X^T @ dY
      // X is [M, K], dY is [M, N] => dW is [K, N]
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
    if (transposeB) {
      gradInput = await recordMatmul(
        recorder,
        gradOutput,
        weight,
        M,
        K,
        N,
        { transposeB: false, role: 'bwd_grad_input' }
      );
    } else {
      gradInput = await recordMatmul(
        recorder,
        gradOutput,
        weight,
        M,
        K,
        N,
        { transposeB: true, role: 'bwd_grad_input' }
      );
    }
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
