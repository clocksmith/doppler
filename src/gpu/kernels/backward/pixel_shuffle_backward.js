import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runPixelShuffleBackward(gradOutput, options = {}) {
  const { outChannels, outHeight, outWidth, gridWidth, gridHeight, patchSize, patchChannels, outputBuffer = null } = options;
  if (!outChannels || !outHeight || !outWidth || !gridWidth || !gridHeight || !patchSize || !patchChannels) {
    throw new Error('pixel_shuffle backward requires all dimensions');
  }

  const device = gradOutput.buffer.device;
  const totalInputElements = gridWidth * gridHeight * patchChannels;
  const outputSize = totalInputElements * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'pixel_shuffle_backward_output');

  const pipeline = await createPipeline('pixel_shuffle_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'pixel_shuffle_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, outChannels, true);
      view.setUint32(4, outHeight, true);
      view.setUint32(8, outWidth, true);
      view.setUint32(12, gridWidth, true);
      view.setUint32(16, gridHeight, true);
      view.setUint32(20, patchSize, true);
      view.setUint32(24, patchChannels, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'pixel_shuffle_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(totalInputElements / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'pixel_shuffle_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, 'f32', [gridHeight * gridWidth, patchChannels], 'pixel_shuffle_backward_output');
}

export async function recordPixelShuffleBackward(recorder, gradOutput, options = {}) {
  const { outChannels, outHeight, outWidth, gridWidth, gridHeight, patchSize, patchChannels, outputBuffer = null } = options;
  const totalInputElements = gridWidth * gridHeight * patchChannels;
  const outputSize = totalInputElements * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'pixel_shuffle_backward_output');

  const pipeline = await createPipeline('pixel_shuffle_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'pixel_shuffle_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, outChannels, true);
      view.setUint32(4, outHeight, true);
      view.setUint32(8, outWidth, true);
      view.setUint32(12, gridWidth, true);
      view.setUint32(16, gridHeight, true);
      view.setUint32(20, patchSize, true);
      view.setUint32(24, patchChannels, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'pixel_shuffle_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(totalInputElements / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'pixel_shuffle_backward');

  return createTensor(outputBuf, 'f32', [gridHeight * gridWidth, patchChannels], 'pixel_shuffle_backward_output');
}
