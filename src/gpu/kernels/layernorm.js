
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { getBuffer } from '../weight-buffer.js';
import { resolveNormWeightDtype } from './rmsnorm.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

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

function normalizeNormWeightDtype(dtype) {
  if (typeof dtype !== 'string') return null;
  const normalized = dtype.toLowerCase();
  return normalized === 'f16' || normalized === 'f32' ? normalized : null;
}

function resolveLayerNormParamDtype(weight, bias, hiddenSize, explicitDtype) {
  const weightDtype = normalizeNormWeightDtype(explicitDtype)
    ?? resolveNormWeightDtype(weight, hiddenSize);
  const biasDtype = resolveNormWeightDtype(bias, hiddenSize);
  if (weightDtype !== biasDtype) {
    throw new Error(
      `[layernorm] weight/bias dtype mismatch: weight=${weightDtype}, bias=${biasDtype}.`
    );
  }
  return weightDtype;
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
  const paramDtype = resolveLayerNormParamDtype(
    weight,
    bias,
    inferredHiddenSize,
    options.normWeightDtype
  );
  const weightBuffer = getBuffer(weight);
  const biasBuffer = getBuffer(bias);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');
  const ownedOutput = outputBuffer ? null : outputBuf;

  try {
    await unifiedKernelWrapper(
      'layernorm',
      null,
      variant,
      [input, weightBuffer, biasBuffer, outputBuf],
      { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps },
      batchSize,
      { PARAMS_IS_F16: paramDtype === 'f16' }
    );

    return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
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
  const paramDtype = resolveLayerNormParamDtype(
    weight,
    bias,
    inferredHiddenSize,
    options.normWeightDtype
  );
  const weightBuffer = getBuffer(weight);
  const biasBuffer = getBuffer(bias);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');
  const ownedOutput = outputBuffer ? null : outputBuf;

  try {
    await unifiedKernelWrapper(
      'layernorm',
      recorder,
      variant,
      [input, weightBuffer, biasBuffer, outputBuf],
      { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps },
      batchSize,
      { PARAMS_IS_F16: paramDtype === 'f16' }
    );

    return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
}
