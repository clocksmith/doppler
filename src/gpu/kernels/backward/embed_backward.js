import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

export async function runEmbedBackward(input, gradOutput, options = {}) {
  void input;
  const device = getDevice();
  const { count, outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'embed_backward_output');

  const pipeline = await createPipeline('embed_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'embed_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'embed_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'embed_backward');
  uniformBuffer.destroy();

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'embed_backward_output');
}

export async function recordEmbedBackward(recorder, input, gradOutput, options = {}) {
  void input;
  const { count, outputBuffer = null } = options;
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'embed_backward_output');

  const pipeline = await createPipeline('embed_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'embed_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'embed_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'embed_backward');

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], 'embed_backward_output');
}
