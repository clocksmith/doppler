import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runBiasAddBackward(gradOutput, options = {}) {
  const { numTokens, dim, outputBuffer = null } = options;
  if (!numTokens || !dim) {
    throw new Error('bias_add backward requires numTokens and dim');
  }

  const device = gradOutput.buffer.device;
  const outputSize = dim * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'bias_add_backward_output');

  const pipeline = await createPipeline('bias_add_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'bias_add_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'bias_add_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(dim / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'bias_add_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, 'f32', [dim], 'bias_add_backward_output');
}

export async function recordBiasAddBackward(recorder, gradOutput, options = {}) {
  const { numTokens, dim, outputBuffer = null } = options;
  if (!numTokens || !dim) {
    throw new Error('bias_add backward requires numTokens and dim');
  }

  const outputSize = dim * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'bias_add_backward_output');

  const pipeline = await createPipeline('bias_add_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'bias_add_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'bias_add_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(dim / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'bias_add_backward');

  return createTensor(outputBuf, 'f32', [dim], 'bias_add_backward_output');
}
