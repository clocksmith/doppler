import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { getDevice } from '../../device.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runScaleBackward(input, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const count = options.count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const device = getDevice();
  const { outputBuffer = null, scale } = options;

  if (scale == null) {
    throw new Error('scale backward requires scale');
  }

  const outputBuf = outputBuffer || acquireBuffer(count * bytesPerElement, undefined, 'scale_backward_output');
  const pipeline = await createPipeline('scale_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'scale_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, count, true);
      view.setFloat32(4, scale, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'scale_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  dispatch(device, pipeline, bindGroup, Math.ceil(count / WORKGROUP_SIZES.DEFAULT), 'scale_backward');
  uniformBuffer.destroy();

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'scale_backward_output');
}

export async function recordScaleBackward(recorder, input, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const count = options.count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const { outputBuffer = null, scale } = options;

  if (scale == null) {
    throw new Error('scale backward requires scale');
  }

  const device = recorder.device;
  const outputBuf = outputBuffer || acquireBuffer(count * bytesPerElement, undefined, 'scale_backward_output');
  const pipeline = await createPipeline('scale_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'scale_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, count, true);
      view.setFloat32(4, scale, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'scale_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, Math.ceil(count / WORKGROUP_SIZES.DEFAULT), 'scale_backward');

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'scale_backward_output');
}
