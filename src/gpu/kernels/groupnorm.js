import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { getBuffer } from '../weight-buffer.js';

function selectGroupNormVariant(stage, isF16) {
  return selectRuleValue('groupnorm', stage, { isF16 });
}

function validateOptions(options) {
  const { channels, height, width, numGroups, eps } = options;
  if (!Number.isFinite(channels) || !Number.isFinite(height) || !Number.isFinite(width)) {
    throw new Error('GroupNorm requires channels/height/width.');
  }
  if (!Number.isFinite(numGroups) || numGroups <= 0) {
    throw new Error('GroupNorm requires numGroups > 0.');
  }
  if (!Number.isFinite(eps)) {
    throw new Error('GroupNorm requires eps.');
  }
}

async function _groupNorm(target, input, weight, bias, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  validateOptions(options);

  const { channels, height, width, numGroups, eps, outputBuffer = null } = options;
  const isF16 = input.dtype === 'f16';
  const statsVariant = selectGroupNormVariant('statsVariant', isF16);
  const applyVariant = selectGroupNormVariant('applyVariant', isF16);

  const uniforms = {
    channels,
    height,
    width,
    num_groups: numGroups,
    eps,
    _pad0: 0,
    _pad1: 0,
    _pad2: 0,
  };

  const statsSize = numGroups * 2 * 4;
  const statsBuffer = acquireBuffer(statsSize, undefined, 'groupnorm_stats');

  await unifiedKernelWrapper(
    'groupnorm_stats',
    target,
    statsVariant,
    [input, statsBuffer],
    uniforms,
    numGroups
  );

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * height * width * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'groupnorm_output');

  const weightBuffer = getBuffer(weight);
  const biasBuffer = getBuffer(bias);

  const total = channels * height * width;
  const workgroups = Math.ceil(total / 256);

  await unifiedKernelWrapper(
    'groupnorm_apply',
    target,
    applyVariant,
    [input, statsBuffer, weightBuffer, biasBuffer, output],
    uniforms,
    workgroups
  );

  if (recorder) {
    recorder.trackTemporaryBuffer(statsBuffer);
  } else {
    releaseBuffer(statsBuffer);
  }

  return createTensor(output, input.dtype, [channels, height, width], 'groupnorm_output');
}

export async function runGroupNorm(input, weight, bias, options = {}) {
  return _groupNorm(null, input, weight, bias, options);
}

export async function recordGroupNorm(recorder, input, weight, bias, options = {}) {
  return _groupNorm(recorder, input, weight, bias, options);
}

