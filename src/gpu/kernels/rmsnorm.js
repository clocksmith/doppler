

import { getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { unifiedKernelWrapper } from './utils.js';

function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}

export function selectRMSNormKernel(options = {}, isF16 = false) {
  const { residual = null, hiddenSize = null } = options;
  const { smallThreshold } = getKernelThresholds().rmsnorm;
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const isSmall = hiddenSize !== null && hiddenSize <= smallThreshold;
  return selectRuleValue(
    'rmsnorm',
    'variant',
    { isF16, residual: !!residual, hasSubgroups, isSmall }
  );
}

export async function runRMSNorm(
  input,
  weight,
  eps,
  options = {}
) {
  const { batchSize = 1, hiddenSize, residual = null, outputBuffer = null, rmsNormWeightOffset = false } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectRMSNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

  // Placeholder for residual if not present
  const residualBuf = residual?.buffer || null;

  await unifiedKernelWrapper(
    'rmsnorm',
    null,
    variant,
    [input, weight, outputBuf, residualBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps, has_residual: residual ? 1 : 0 },
    batchSize,
    { RMS_NORM_OFFSET: rmsNormWeightOffset }
  );

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
}

export async function recordRMSNorm(
  recorder,
  input,
  weight,
  eps,
  options = {}
) {
  const { batchSize = 1, hiddenSize = null, residual = null, outputBuffer = null, rmsNormWeightOffset = false } = options;
  const isF16 = input.dtype === 'f16';
  const variant = selectRMSNormKernel(options, isF16);
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

  const residualBuf = residual?.buffer || null;

  await unifiedKernelWrapper(
    'rmsnorm',
    recorder,
    variant,
    [input, weight, outputBuf, residualBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps, has_residual: residual ? 1 : 0 },
    batchSize,
    { RMS_NORM_OFFSET: rmsNormWeightOffset }
  );

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
}
