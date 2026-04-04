

import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runLayerNorm } from '../../../gpu/kernels/layernorm.js';
import { runMatmul } from '../../../gpu/kernels/matmul.js';
import { runGelu } from '../../../gpu/kernels/gelu.js';
import { runBiasAdd, runResidualAdd } from '../../../gpu/kernels/residual.js';

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
  const inputTensor = createTensor(a, 'f32', [M, K], 'vision_matmul_input');
  const projected = await runMatmul(inputTensor, b, M, N, K, {
    outputDtype: 'f32',
  });
  if (!bias) {
    return projected.buffer;
  }

  const biasTensor = createTensor(bias, 'f32', [N], 'vision_matmul_bias');
  const biased = await runBiasAdd(projected, biasTensor, M, N);
  return biased.buffer;
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
