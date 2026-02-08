import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runGroupNormBackward(input, weight, gradOutput, options = {}) {
  const { channels, height, width, numGroups, eps = 1e-5, outputBuffer = null } = options;
  if (!channels || !height || !width || !numGroups) {
    throw new Error('groupnorm backward requires all dimensions');
  }

  const device = gradOutput.buffer.device;
  const outputSize = channels * height * width * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'groupnorm_backward_output');

  const pipeline = await createPipeline('groupnorm_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'groupnorm_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, numGroups, true);
      view.setFloat32(16, eps, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'groupnorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numGroups;
  dispatch(device, pipeline, bindGroup, workgroups, 'groupnorm_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, 'f32', [channels, height, width], 'groupnorm_backward_output');
}

export async function recordGroupNormBackward(recorder, input, weight, gradOutput, options = {}) {
  const { channels, height, width, numGroups, eps = 1e-5, outputBuffer = null } = options;
  const outputSize = channels * height * width * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'groupnorm_backward_output');

  const pipeline = await createPipeline('groupnorm_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'groupnorm_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, numGroups, true);
      view.setFloat32(16, eps, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'groupnorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numGroups;
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'groupnorm_backward');

  return createTensor(outputBuf, 'f32', [channels, height, width], 'groupnorm_backward_output');
}
