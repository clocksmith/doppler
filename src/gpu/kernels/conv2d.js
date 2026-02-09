
import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectConv2DVariant(isF16) {
  return selectRuleValue('conv2d', 'variant', { isF16 });
}

async function _conv2d(target, input, weight, bias, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const device = target?.device || getDevice();
  const {
    inChannels, outChannels, height, width,
    kernelH, kernelW, stride = 1, pad = 0,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(inChannels) || !Number.isFinite(outChannels) ||
      !Number.isFinite(height) || !Number.isFinite(width) ||
      !Number.isFinite(kernelH) || !Number.isFinite(kernelW)) {
    throw new Error('Conv2D requires explicit dimensions.');
  }

  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  if (outHeight <= 0 || outWidth <= 0) {
    throw new Error(`Conv2D invalid output size: ${outHeight}x${outWidth}`);
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectConv2DVariant(isF16);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'conv2d_output');

  const weightBuffer = getBuffer(weight);
  let biasBuffer = getBuffer(bias);
  let tempBias = null;
  if (!biasBuffer) {
    const biasSize = outChannels * bytesPerElement;
    tempBias = acquireBuffer(biasSize, undefined, 'conv2d_bias_zero');
    biasBuffer = tempBias;
    const paddedSize = Math.ceil(biasSize / 4) * 4;
    device.queue.writeBuffer(biasBuffer, 0, new Uint8Array(paddedSize));
  }

  await unifiedKernelWrapper(
    'conv2d', target, variant,
    [input, weightBuffer, biasBuffer, output],
    {
      in_channels: inChannels, out_channels: outChannels,
      height, width, out_height: outHeight, out_width: outWidth,
      kernel_h: kernelH, kernel_w: kernelW,
      stride, pad, _pad0: 0, _pad1: 0,
    },
    Math.ceil((outChannels * outHeight * outWidth) / 256)
  );

  if (tempBias) {
    if (recorder) {
      recorder.trackTemporaryBuffer(tempBias);
    } else {
      releaseBuffer(tempBias);
    }
  }

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'conv2d_output');
}

export async function runConv2D(input, weight, bias, options = {}) {
  return _conv2d(null, input, weight, bias, options);
}

export async function recordConv2D(recorder, input, weight, bias, options = {}) {
  return _conv2d(recorder, input, weight, bias, options);
}
