import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectReluVariant(dtype) {
  return selectRuleValue('relu', 'variant', { dtype });
}

function resolveCount(input, countOverride) {
  if (Number.isFinite(countOverride) && countOverride > 0) {
    return Math.floor(countOverride);
  }
  if (Array.isArray(input.shape) && input.shape.length > 0) {
    return input.shape.reduce((acc, value) => acc * value, 1);
  }
  return Math.floor(input.buffer.size / dtypeBytes(input.dtype));
}

function planReluDispatch(target, size) {
  const device = target?.device;
  const maxPerDim = Number.isFinite(device?.limits?.maxComputeWorkgroupsPerDimension)
    ? device.limits.maxComputeWorkgroupsPerDimension
    : 65535;
  const dispatchStride = Math.min(size, maxPerDim * WORKGROUP_SIZES.DEFAULT);
  return {
    dispatchStride,
    workgroups: [Math.ceil(dispatchStride / WORKGROUP_SIZES.DEFAULT), 1, 1],
  };
}

async function _relu(target, input, options = {}) {
  const { count = null, outputBuffer = null } = options;
  const size = resolveCount(input, count);
  const variant = selectReluVariant(input.dtype);
  const output = outputBuffer || acquireBuffer(size * dtypeBytes(input.dtype), undefined, 'relu_output');
  const ownedOutput = outputBuffer ? null : output;
  const dispatchPlan = planReluDispatch(target, size);

  try {
    await unifiedKernelWrapper(
      'relu',
      target,
      variant,
      [input, output],
      { size, _pad0: dispatchPlan.dispatchStride, _pad1: 0, _pad2: 0 },
      dispatchPlan.workgroups
    );

    return createTensor(output, input.dtype, [...input.shape], 'relu_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
}

export async function runReLU(input, options = {}) {
  return _relu(null, input, options);
}

export async function recordReLU(recorder, input, options = {}) {
  return _relu(recorder, input, options);
}
