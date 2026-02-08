import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runUpsample2DBackward(gradOutput, options = {}) {
  const { channels, inHeight, inWidth, outHeight, outWidth, scale, outputBuffer = null } = options;
  if (!channels || !inHeight || !inWidth || !outHeight || !outWidth || !scale) {
    throw new Error('upsample2d backward requires channels, inHeight, inWidth, outHeight, outWidth, and scale');
  }

  const device = gradOutput.buffer.device;
  const outputSize = channels * inHeight * inWidth * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'upsample2d_backward_output');

  const pipeline = await createPipeline('upsample2d_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'upsample2d_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, inHeight, true);
      view.setUint32(8, inWidth, true);
      view.setUint32(12, outHeight, true);
      view.setUint32(16, outWidth, true);
      view.setUint32(20, scale, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'upsample2d_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const totalInElements = channels * inHeight * inWidth;
  const workgroups = Math.ceil(totalInElements / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'upsample2d_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, 'f32', [channels, inHeight, inWidth], 'upsample2d_backward_output');
}

export async function recordUpsample2DBackward(recorder, gradOutput, options = {}) {
  const { channels, inHeight, inWidth, outHeight, outWidth, scale, outputBuffer = null } = options;
  const outputSize = channels * inHeight * inWidth * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'upsample2d_backward_output');

  const pipeline = await createPipeline('upsample2d_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'upsample2d_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, inHeight, true);
      view.setUint32(8, inWidth, true);
      view.setUint32(12, outHeight, true);
      view.setUint32(16, outWidth, true);
      view.setUint32(20, scale, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'upsample2d_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const totalInElements = channels * inHeight * inWidth;
  const workgroups = Math.ceil(totalInElements / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'upsample2d_backward');

  return createTensor(outputBuf, 'f32', [channels, inHeight, inWidth], 'upsample2d_backward_output');
}
