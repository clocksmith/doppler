import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';
import { getKernelConfig } from './kernel-configs.js';
import { getDevice } from '../device.js';

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
    !Number.isSafeInteger(inChannels) || inChannels < 1 ||
    !Number.isSafeInteger(height) || height < 1 ||
    !Number.isSafeInteger(width) || width < 1 ||
    !Number.isSafeInteger(repeats) || repeats < 1
  ) {
    throw new Error('RepeatChannels requires positive integer inChannels, height, width, and repeats.');
  }

  const outChannels = inChannels * repeats;
  const count = outChannels * height * width;
  if (!Number.isSafeInteger(count) || count > 0xFFFFFFFF) {
    throw new Error('RepeatChannels output element count must fit a u32.');
  }
  const variant = selectRepeatChannelsVariant(input.dtype);
  const config = getKernelConfig('repeat_channels', variant);
  const threads = config.wgslOverrides?.WORKGROUP_SIZE ?? config.workgroupSize[0];
  const limit = (target?.device ?? getDevice())?.limits?.maxComputeWorkgroupsPerDimension;
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new Error('RepeatChannels requires an explicit device workgroup limit.');
  }
  const groups = Math.ceil(count / threads);
  const x = Math.min(groups, limit), y = Math.ceil(groups / x);
  if (y > limit) throw new Error('RepeatChannels exceeds two-dimensional device workgroup capacity.');
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = count * bytesPerElement;
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
      [x, y, 1]
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
