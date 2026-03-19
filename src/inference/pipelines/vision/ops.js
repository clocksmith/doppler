

import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { runLayerNorm } from '../../../gpu/kernels/layernorm.js';
import { dispatchMatmul } from '../../../gpu/kernels/matmul-dispatch.js';
import { runGelu } from '../../../gpu/kernels/gelu.js';
import { runResidualAdd } from '../../../gpu/kernels/residual.js';

/**
 * Layer norm on GPU.
 * @param {GPUBuffer} input   [seqLen, hiddenSize]
 * @param {GPUBuffer} weight  [hiddenSize]
 * @param {GPUBuffer} bias    [hiddenSize] or null
 * @param {{ seqLen: number, hiddenSize: number, eps: number }} opts
 * @returns {Promise<GPUBuffer>}
 */
export async function doLayerNorm(input, weight, bias, opts) {
  const { seqLen, hiddenSize, eps } = opts;
  const outputSize = seqLen * hiddenSize * 4;
  const output = acquireBuffer(outputSize, 'vision-layernorm');
  await runLayerNorm({
    input,
    weight,
    bias: bias || null,
    output,
    seqLen,
    hiddenSize,
    eps,
  });
  return output;
}

/**
 * Matrix multiply on GPU.
 * @param {GPUBuffer} a  [M, K]
 * @param {GPUBuffer} b  [K, N]
 * @param {{ M: number, K: number, N: number, bias?: GPUBuffer }} opts
 * @returns {Promise<GPUBuffer>}
 */
export async function doMatmul(a, b, opts) {
  const { M, K, N, bias } = opts;
  const outputSize = M * N * 4;
  const output = acquireBuffer(outputSize, 'vision-matmul');
  await dispatchMatmul({
    a, b, output,
    M, K, N,
    bias: bias || null,
  });
  return output;
}

/**
 * GELU activation on GPU.
 * @param {GPUBuffer} input   Flat buffer
 * @param {{ count: number }} opts  Total element count
 * @returns {Promise<GPUBuffer>}
 */
export async function doGelu(input, opts) {
  const { count } = opts;
  const output = acquireBuffer(count * 4, 'vision-gelu');
  await runGelu({ input, output, count });
  return output;
}

/**
 * Element-wise residual add on GPU.
 * @param {GPUBuffer} a
 * @param {GPUBuffer} b
 * @param {{ count: number }} opts
 * @returns {Promise<GPUBuffer>}
 */
export async function doResidualAdd(a, b, opts) {
  const { count } = opts;
  const output = acquireBuffer(count * 4, 'vision-residual');
  await runResidualAdd({ a, b, output, count });
  return output;
}
