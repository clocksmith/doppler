import { runMatmul, recordMatmul } from '../matmul.js';
import { runTranspose, recordTranspose } from '../transpose.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';

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
    if (transposeB) {
      gradInput = await runMatmul(
        gradOutput,
        weight,
        M,
        K,
        N,
        { transposeB: false }
      );
    } else {
      const weightTransposed = await runTranspose(weight, K, N);
      gradInput = await runMatmul(
        gradOutput,
        weightTransposed,
        M,
        K,
        N,
        { transposeB: false }
      );
      releaseBuffer(weightTransposed.buffer);
    }
  }

  if (computeGradWeight) {
    if (transposeB) {
      const gradOutputTransposed = await runTranspose(gradOutput, M, N);
      gradWeight = await runMatmul(
        gradOutputTransposed,
        input,
        N,
        K,
        M,
        { transposeB: false }
      );
      releaseBuffer(gradOutputTransposed.buffer);
    } else {
      const inputTransposed = await runTranspose(input, M, K);
      gradWeight = await runMatmul(
        inputTransposed,
        gradOutput,
        K,
        N,
        M,
        { transposeB: false }
      );
      releaseBuffer(inputTransposed.buffer);
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
      const weightTransposed = await recordTranspose(recorder, weight, K, N);
      recorder.trackTemporaryBuffer(weightTransposed.buffer);
      gradInput = await recordMatmul(
        recorder,
        gradOutput,
        weightTransposed,
        M,
        K,
        N,
        { transposeB: false, role: 'bwd_grad_input' }
      );
    }
  }

  if (computeGradWeight) {
    if (transposeB) {
      const gradOutputTransposed = await recordTranspose(recorder, gradOutput, M, N);
      recorder.trackTemporaryBuffer(gradOutputTransposed.buffer);
      gradWeight = await recordMatmul(
        recorder,
        gradOutputTransposed,
        input,
        N,
        K,
        M,
        { transposeB: false, role: 'bwd_grad_weight' }
      );
    } else {
      const inputTransposed = await recordTranspose(recorder, input, M, K);
      recorder.trackTemporaryBuffer(inputTransposed.buffer);
      gradWeight = await recordMatmul(
        recorder,
        inputTransposed,
        gradOutput,
        K,
        N,
        M,
        { transposeB: false, role: 'bwd_grad_weight' }
      );
    }
  }

  return { gradInput, gradWeight };
}
