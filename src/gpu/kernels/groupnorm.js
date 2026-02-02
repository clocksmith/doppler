import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
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

export async function runGroupNorm(
  input,
  weight,
  bias,
  options = {}
) {
  const device = getDevice();
  validateOptions(options);

  const { channels, height, width, numGroups, eps, outputBuffer = null } = options;
  const isF16 = input.dtype === 'f16';
  const statsVariant = selectGroupNormVariant('statsVariant', isF16);
  const applyVariant = selectGroupNormVariant('applyVariant', isF16);

  const statsPipeline = await getPipelineFast('groupnorm_stats', statsVariant);
  const applyPipeline = await getPipelineFast('groupnorm_apply', applyVariant);

  const uniformBuffer = createUniformBufferWithView(
    'groupnorm_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, numGroups, true);
      view.setFloat32(16, eps, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );

  const statsSize = numGroups * 2 * 4;
  const statsBuffer = acquireBuffer(statsSize, undefined, 'groupnorm_stats');

  const statsBindGroup = device.createBindGroup({
    label: 'groupnorm_stats_bind_group',
    layout: statsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: statsBuffer } },
    ],
  });

  dispatch(device, statsPipeline, statsBindGroup, numGroups, 'groupnorm_stats');

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * height * width * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'groupnorm_output');

  const weightBuffer = getBuffer(weight);
  const biasBuffer = getBuffer(bias);

  const applyBindGroup = device.createBindGroup({
    label: 'groupnorm_apply_bind_group',
    layout: applyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: statsBuffer } },
      { binding: 3, resource: { buffer: weightBuffer } },
      { binding: 4, resource: { buffer: biasBuffer } },
      { binding: 5, resource: { buffer: output } },
    ],
  });

  const total = channels * height * width;
  const workgroups = Math.ceil(total / 256);
  dispatch(device, applyPipeline, applyBindGroup, workgroups, 'groupnorm_apply');

  uniformBuffer.destroy();
  releaseBuffer(statsBuffer);

  return createTensor(output, input.dtype, [channels, height, width], 'groupnorm_output');
}

export async function recordGroupNorm(
  recorder,
  input,
  weight,
  bias,
  options = {}
) {
  validateOptions(options);

  const { channels, height, width, numGroups, eps, outputBuffer = null } = options;
  const isF16 = input.dtype === 'f16';
  const statsVariant = selectGroupNormVariant('statsVariant', isF16);
  const applyVariant = selectGroupNormVariant('applyVariant', isF16);

  const statsPipeline = await getPipelineFast('groupnorm_stats', statsVariant);
  const applyPipeline = await getPipelineFast('groupnorm_apply', applyVariant);

  const uniformBuffer = createUniformBufferWithView(
    'groupnorm_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, numGroups, true);
      view.setFloat32(16, eps, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder
  );

  const statsSize = numGroups * 2 * 4;
  const statsBuffer = acquireBuffer(statsSize, undefined, 'groupnorm_stats');

  const statsBindGroup = recorder.device.createBindGroup({
    label: 'groupnorm_stats_bind_group',
    layout: statsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: statsBuffer } },
    ],
  });

  recordDispatch(recorder, statsPipeline, statsBindGroup, numGroups, 'groupnorm_stats');

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * height * width * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'groupnorm_output');

  const weightBuffer = getBuffer(weight);
  const biasBuffer = getBuffer(bias);

  const applyBindGroup = recorder.device.createBindGroup({
    label: 'groupnorm_apply_bind_group',
    layout: applyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: statsBuffer } },
      { binding: 3, resource: { buffer: weightBuffer } },
      { binding: 4, resource: { buffer: biasBuffer } },
      { binding: 5, resource: { buffer: output } },
    ],
  });

  const total = channels * height * width;
  const workgroups = Math.ceil(total / 256);
  recordDispatch(recorder, applyPipeline, applyBindGroup, workgroups, 'groupnorm_apply');

  recorder.trackTemporaryBuffer(statsBuffer);

  return createTensor(output, input.dtype, [channels, height, width], 'groupnorm_output');
}
