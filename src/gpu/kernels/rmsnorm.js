

import { getKernelCapabilities } from '../device.js';
import { acquireBuffer, getBufferRequestedSize } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getKernelThresholds, padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';
import { getBuffer, getWeightDtype, getBufferDtype } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';

function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}

function normalizeNormWeightDtype(dtype) {
  if (typeof dtype !== 'string') return null;
  const value = dtype.toLowerCase();
  if (value === 'f16' || value === 'f32') {
    return value;
  }
  return null;
}

function resolveNormWeightDtype(weight, hiddenSize) {
  const explicitDtype = normalizeNormWeightDtype(getWeightDtype(weight));
  if (explicitDtype) {
    return explicitDtype;
  }

  const weightBuffer = getBuffer(weight);
  const taggedDtype = normalizeNormWeightDtype(getBufferDtype(weightBuffer));
  if (taggedDtype) {
    return taggedDtype;
  }

  const hasGPUBufferType = typeof GPUBuffer !== 'undefined';
  if (!hasGPUBufferType || !(weightBuffer instanceof GPUBuffer) || hiddenSize == null || hiddenSize <= 0) {
    return 'f32';
  }

  const byteSize = getBufferRequestedSize(weightBuffer);
  const f16Bytes = hiddenSize * 2;
  const f32Bytes = hiddenSize * 4;
  const sizeMatchesF16 = byteSize === f16Bytes;
  const sizeMatchesF32 = byteSize === f32Bytes;
  if (sizeMatchesF16 || sizeMatchesF32) {
    return selectLoaderRule('loader', 'weights', 'normWeightDtypeFromSize', {
      sizeMatchesF16,
      sizeMatchesF32,
    });
  }
  return 'f32';
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
  const normWeightBuffer = getBuffer(weight);
  const normWeightDtype = resolveNormWeightDtype(weight, inferredHiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

  // Shader layout always includes the residual binding; when unused, bind a harmless placeholder.
  const residualBuf = residual?.buffer || input.buffer;

  await unifiedKernelWrapper(
    'rmsnorm',
    null,
    variant,
    [input, normWeightBuffer, outputBuf, residualBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps, has_residual: residual ? 1 : 0 },
    batchSize,
    { RMS_NORM_OFFSET: rmsNormWeightOffset, WEIGHT_IS_F16: normWeightDtype === 'f16' }
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
  const normWeightBuffer = getBuffer(weight);
  const normWeightDtype = resolveNormWeightDtype(weight, inferredHiddenSize);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_output');

  const residualBuf = residual?.buffer || input.buffer;

  await unifiedKernelWrapper(
    'rmsnorm',
    recorder,
    variant,
    [input, normWeightBuffer, outputBuf, residualBuf],
    { hidden_size: inferredHiddenSize, num_tokens: batchSize, eps, has_residual: residual ? 1 : 0 },
    batchSize,
    { RMS_NORM_OFFSET: rmsNormWeightOffset, WEIGHT_IS_F16: normWeightDtype === 'f16' }
  );

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'rmsnorm_output');
}
