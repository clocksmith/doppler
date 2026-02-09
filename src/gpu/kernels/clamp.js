import { createTensor } from '../tensor.js';
import { dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';

function resolveCount(tensor, countOverride) {
  if (Number.isFinite(countOverride) && countOverride > 0) {
    return Math.floor(countOverride);
  }
  if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
    return tensor.shape.reduce((acc, value) => acc * value, 1);
  }
  return Math.floor(tensor.buffer.size / dtypeBytes(tensor.dtype));
}

async function _clamp(target, input, minValue, maxValue, options = {}) {
  if (input.dtype !== 'f32') {
    throw new Error(`clamp: unsupported dtype ${input.dtype}.`);
  }

  const { count } = options;
  const inferredCount = resolveCount(input, count);

  await unifiedKernelWrapper(
    'clamp',
    target,
    'default',
    [input],
    { size: inferredCount, min: minValue, max: maxValue },
    Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT)
  );

  return createTensor(input.buffer, input.dtype, [...input.shape], 'clamp_output');
}

export async function runClamp(input, minValue, maxValue, options = {}) {
  return _clamp(null, input, minValue, maxValue, options);
}

export async function recordClamp(recorder, input, minValue, maxValue, options = {}) {
  return _clamp(recorder, input, minValue, maxValue, options);
}
