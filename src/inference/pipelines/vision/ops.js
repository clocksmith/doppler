

import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runLayerNorm } from '../../../gpu/kernels/layernorm.js';
import { runMatmul } from '../../../gpu/kernels/matmul.js';
import { runGeLU } from '../../../gpu/kernels/gelu.js';
import { runBiasAdd, runResidualAdd } from '../../../gpu/kernels/residual.js';

export async function doLayerNorm(input, weight, bias, opts) {
  const { seqLen, hiddenSize, eps } = opts;
  const inputTensor = createTensor(input, 'f32', [seqLen, hiddenSize], 'vision_layernorm_input');
  const outputTensor = await runLayerNorm(
    inputTensor,
    weight,
    bias || null,
    eps,
    {
      batchSize: seqLen,
      hiddenSize,
    }
  );
  return outputTensor.buffer;
}

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

export async function doGelu(input, opts) {
  const { count } = opts;
  const inputTensor = createTensor(input, 'f32', [count], 'vision_gelu_input');
  const outputTensor = await runGeLU(inputTensor, { size: count });
  return outputTensor.buffer;
}

export async function doResidualAdd(a, b, opts) {
  const { count } = opts;
  const aTensor = createTensor(a, 'f32', [count], 'vision_residual_a');
  const bTensor = createTensor(b, 'f32', [count], 'vision_residual_b');
  const outputTensor = await runResidualAdd(aTensor, bTensor, count);
  return outputTensor.buffer;
}
