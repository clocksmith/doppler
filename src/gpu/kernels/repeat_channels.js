import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectRepeatChannelsVariant(dtype) {
  return selectRuleValue('repeatChannels', 'variant', { dtype });
}

async function _repeatChannels(target, input, options = {}) {
  const {
    inChannels,
    height,
    width,
    repeats,
    outputBuffer = null,
  } = options;

  if (
    !Number.isFinite(inChannels) ||
    !Number.isFinite(height) ||
    !Number.isFinite(width) ||
    !Number.isFinite(repeats) ||
    repeats < 1
  ) {
    throw new Error('RepeatChannels requires inChannels, height, width, and repeats.');
  }

  const outChannels = inChannels * repeats;
  const variant = selectRepeatChannelsVariant(input.dtype);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * height * width * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'repeat_channels_output');
  const ownedOutput = outputBuffer ? null : output;

  try {
    await unifiedKernelWrapper(
      'repeat_channels',
      target,
      variant,
      [input, output],
      {
        in_channels: inChannels,
        height,
        width,
        repeats,
        _pad0: 0,
      },
      [Math.ceil((height * width) / WORKGROUP_SIZES.DEFAULT), outChannels, 1]
    );

    return createTensor(output, input.dtype, [outChannels, height, width], 'repeat_channels_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
}

export async function runRepeatChannels(input, options = {}) {
  return _repeatChannels(null, input, options);
}

export async function recordRepeatChannels(recorder, input, options = {}) {
  return _repeatChannels(recorder, input, options);
}
