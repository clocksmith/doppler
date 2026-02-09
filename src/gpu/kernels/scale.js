import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

async function _scale(target, input, scale, options = {}) {
  const { count, outputBuffer = null, inplace = false } = options;

  const bytesPerElement = dtypeBytes(input.dtype);
  const inferredCount = count ?? Math.floor(input.buffer.size / bytesPerElement);
  const variant = selectRuleValue('scale', 'variant', { inplace });

  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = inplace ? input.buffer : (outputBuffer || acquireBuffer(outputSize, undefined, 'scale_output'));

  const bindings = inplace ? [outputBuf, outputBuf] : [input, outputBuf];

  await unifiedKernelWrapper(
    'scale',
    target,
    variant,
    bindings,
    { size: inferredCount, scale },
    Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT)
  );

  return createTensor(outputBuf, input.dtype, [...input.shape], 'scale_output');
}

export async function runScale(input, scale, options = {}) {
  return _scale(null, input, scale, options);
}

export async function recordScale(recorder, input, scale, options = {}) {
  return _scale(recorder, input, scale, options);
}
