import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectDepthwiseConv2DVariant(isF16) {
  return selectRuleValue('depthwiseConv2d', 'variant', { isF16 });
}

async function _depthwiseConv2D(target, input, weight, bias, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const device = target?.device || getDevice();
  const {
    channels,
    height,
    width,
    kernelH,
    kernelW,
    stride = 1,
    pad = 0,
    outputBuffer = null,
  } = options;

  if (
    !Number.isFinite(channels) ||
    !Number.isFinite(height) ||
    !Number.isFinite(width) ||
    !Number.isFinite(kernelH) ||
    !Number.isFinite(kernelW)
  ) {
    throw new Error('DepthwiseConv2D requires explicit dimensions.');
  }

  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  if (outHeight <= 0 || outWidth <= 0) {
    throw new Error(`DepthwiseConv2D invalid output size: ${outHeight}x${outWidth}`);
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectDepthwiseConv2DVariant(isF16);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'depthwise_conv2d_output');
  const outSpatial = outHeight * outWidth;

  const weightBuffer = getBuffer(weight);
  let biasBuffer = getBuffer(bias);
  let tempBias = null;
  if (!biasBuffer) {
    const biasSize = channels * bytesPerElement;
    tempBias = acquireBuffer(biasSize, undefined, 'depthwise_conv2d_bias_zero');
    biasBuffer = tempBias;
    const paddedSize = Math.ceil(biasSize / 4) * 4;
    device.queue.writeBuffer(biasBuffer, 0, new Uint8Array(paddedSize));
  }

  await unifiedKernelWrapper(
    'depthwise_conv2d',
    target,
    variant,
    [input, weightBuffer, biasBuffer, output],
    {
      channels,
      height,
      width,
      out_height: outHeight,
      out_width: outWidth,
      kernel_h: kernelH,
      kernel_w: kernelW,
      stride,
      pad,
      _pad0: 0,
      _pad1: 0,
    },
    [Math.ceil(outSpatial / WORKGROUP_SIZES.DEFAULT), channels, 1]
  );

  if (tempBias) {
    if (recorder) {
      recorder.trackTemporaryBuffer(tempBias);
    } else {
      releaseBuffer(tempBias);
    }
  }

  return createTensor(output, input.dtype, [channels, outHeight, outWidth], 'depthwise_conv2d_output');
}

export async function runDepthwiseConv2D(input, weight, bias, options = {}) {
  return _depthwiseConv2D(null, input, weight, bias, options);
}

export async function recordDepthwiseConv2D(recorder, input, weight, bias, options = {}) {
  return _depthwiseConv2D(recorder, input, weight, bias, options);
}
