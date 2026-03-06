import { acquireBuffer } from '../../memory/buffer-pool.js';
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

async function _relu(target, input, options = {}) {
  const { count = null, outputBuffer = null } = options;
  const size = resolveCount(input, count);
  const variant = selectReluVariant(input.dtype);
  const output = outputBuffer || acquireBuffer(size * dtypeBytes(input.dtype), undefined, 'relu_output');

  await unifiedKernelWrapper(
    'relu',
    target,
    variant,
    [input, output],
    { size, _pad0: 0, _pad1: 0, _pad2: 0 },
    Math.ceil(size / WORKGROUP_SIZES.DEFAULT)
  );

  return createTensor(output, input.dtype, [...input.shape], 'relu_output');
}

export async function runReLU(input, options = {}) {
  return _relu(null, input, options);
}

export async function recordReLU(recorder, input, options = {}) {
  return _relu(recorder, input, options);
}
