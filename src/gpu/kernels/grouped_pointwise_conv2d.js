import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectGroupedPointwiseConv2DVariant(isF16) {
  return selectRuleValue('groupedPointwiseConv2d', 'variant', { isF16 });
}

async function _groupedPointwiseConv2D(target, input, weight, bias, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const device = target?.device || getDevice();
  const {
    inChannels,
    outChannels,
    height,
    width,
    groups,
    outputBuffer = null,
  } = options;

  if (
    !Number.isFinite(inChannels) ||
    !Number.isFinite(outChannels) ||
    !Number.isFinite(height) ||
    !Number.isFinite(width) ||
    !Number.isFinite(groups)
  ) {
    throw new Error('GroupedPointwiseConv2D requires explicit dimensions.');
  }
  if (groups <= 0 || inChannels % groups !== 0 || outChannels % groups !== 0) {
    throw new Error(
      `GroupedPointwiseConv2D requires inChannels/outChannels divisible by groups. Got ${inChannels}/${outChannels}/${groups}.`
    );
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectGroupedPointwiseConv2DVariant(isF16);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * height * width * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'grouped_pointwise_conv2d_output');
  const spatial = height * width;

  const weightBuffer = getBuffer(weight);
  let biasBuffer = getBuffer(bias);
  let tempBias = null;
  if (!biasBuffer) {
    const biasSize = outChannels * bytesPerElement;
    tempBias = acquireBuffer(biasSize, undefined, 'grouped_pointwise_conv2d_bias_zero');
    biasBuffer = tempBias;
    const paddedSize = Math.ceil(biasSize / 4) * 4;
    device.queue.writeBuffer(biasBuffer, 0, new Uint8Array(paddedSize));
  }

  try {
    await unifiedKernelWrapper(
      'grouped_pointwise_conv2d',
      target,
      variant,
      [input, weightBuffer, biasBuffer, output],
      {
        in_channels: inChannels,
        out_channels: outChannels,
        height,
        width,
        groups,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      [Math.ceil(spatial / WORKGROUP_SIZES.DEFAULT), outChannels, 1]
    );

    if (tempBias) {
      if (recorder) {
        recorder.trackTemporaryBuffer(tempBias);
      } else {
        releaseBuffer(tempBias);
      }
    }

    return createTensor(output, input.dtype, [outChannels, height, width], 'grouped_pointwise_conv2d_output');
  } catch (error) {
    if (tempBias) {
      releaseBuffer(tempBias);
    }
    if (!outputBuffer) {
      releaseBuffer(output);
    }
    throw error;
  }
}

export async function runGroupedPointwiseConv2D(input, weight, bias, options = {}) {
  return _groupedPointwiseConv2D(null, input, weight, bias, options);
}

export async function recordGroupedPointwiseConv2D(recorder, input, weight, bias, options = {}) {
  return _groupedPointwiseConv2D(recorder, input, weight, bias, options);
}
