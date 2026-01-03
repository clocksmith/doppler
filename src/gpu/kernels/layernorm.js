
import { getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';
import { unifiedKernelWrapper } from './utils.js';

function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}

export function selectLayerNormKernel(options = {}, isF16 = false) {
  return selectRuleValue('layernorm', 'variant', { isF16 });
}

export async function runLayerNorm(
  input,
  weight,
  bias,
  eps,
  options = {}
) {
  const { batchSize = 1, hiddenSize = null, outputBuffer = null } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectLayerNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');

  await unifiedKernelWrapper(
    'layernorm',
    null,
    variant,
    [input, weight, bias, outputBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps },
    batchSize
  );

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
}

export async function recordLayerNorm(
  recorder,
  input,
  weight,
  bias,
  eps,
  options = {}
) {
  const { batchSize = 1, hiddenSize = null, outputBuffer = null } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectLayerNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');

  await unifiedKernelWrapper(
    'layernorm',
    recorder,
    variant,
    [input, weight, bias, outputBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps },
    batchSize
  );

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
}
