import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';

async function _transpose(target, input, rows, cols, options = {}) {
  const { outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = rows * cols * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'transpose_output');

  await unifiedKernelWrapper(
    'transpose',
    target,
    'default',
    [input, outputBuf],
    { rows, cols },
    Math.ceil((rows * cols) / WORKGROUP_SIZES.DEFAULT)
  );

  return createTensor(outputBuf, input.dtype, [cols, rows], 'transpose_output');
}

export async function runTranspose(input, rows, cols, options = {}) {
  return _transpose(null, input, rows, cols, options);
}

export async function recordTranspose(recorder, input, rows, cols, options = {}) {
  return _transpose(recorder, input, rows, cols, options);
}
