import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';

export async function runTranspose(input, rows, cols, options = {}) {
  const device = getDevice();
  const { outputBuffer = null } = options;

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = rows * cols * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'transpose_output');

  const pipeline = await createPipeline('transpose', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'transpose_uniforms',
    16,
    (view) => {
      view.setUint32(0, rows, true);
      view.setUint32(4, cols, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'transpose_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil((rows * cols) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'transpose');

  uniformBuffer.destroy();

  return createTensor(outputBuf, input.dtype, [cols, rows], 'transpose_output');
}

export async function recordTranspose(recorder, input, rows, cols, options = {}) {
  const device = recorder.device;
  const { outputBuffer = null } = options;

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = rows * cols * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'transpose_output');

  const pipeline = await createPipeline('transpose', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'transpose_uniforms',
    16,
    (view) => {
      view.setUint32(0, rows, true);
      view.setUint32(4, cols, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'transpose_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil((rows * cols) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'transpose');

  return createTensor(outputBuf, input.dtype, [cols, rows], 'transpose_output');
}
