import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

export async function runScaleBackward(input, gradOutput, options = {}) {
  void input;
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }

  const device = getDevice();
  const { count, outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scale_backward_output');
  const pipeline = await createPipeline('scale_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'scale_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
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

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'scale_backward');
  uniformBuffer.destroy();

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'scale_backward_output');
}

export async function recordScaleBackward(recorder, input, gradOutput, options = {}) {
  void input;
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }

  const { count, outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scale_backward_output');
  const pipeline = await createPipeline('scale_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'scale_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'scale_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'scale_backward');

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'scale_backward_output');
}
