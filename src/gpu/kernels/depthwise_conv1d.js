
import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import { WORKGROUP_SIZES } from './constants.js';

async function _depthwiseConv1d(target, input, weight, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const { channels, length, kernelSize, outputBuffer = null } = options;

  if (!Number.isFinite(channels) || !Number.isFinite(length) || !Number.isFinite(kernelSize)) {
    throw new Error('DepthwiseConv1D requires explicit channels, length, and kernelSize.');
  }

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * length * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'depthwise_conv1d_output');

  const weightBuffer = getBuffer(weight);

  try {
    await unifiedKernelWrapper(
      'depthwise_conv1d', target, 'depthwise_conv1d',
      [input, weightBuffer, output],
      {
        channels,
        length,
        kernel_size: kernelSize,
        _pad0: 0,
      },
      [Math.ceil(length / WORKGROUP_SIZES.DEFAULT), channels, 1]
    );

    return createTensor(output, input.dtype, [channels, length], 'depthwise_conv1d_output');
  } catch (error) {
    if (!outputBuffer) {
      releaseBuffer(output);
    }
    throw error;
  }
}

export async function runDepthwiseConv1D(input, weight, options) {
  return _depthwiseConv1d(null, input, weight, options);
}

export async function recordDepthwiseConv1D(recorder, input, weight, options) {
  return _depthwiseConv1d(recorder, input, weight, options);
}
