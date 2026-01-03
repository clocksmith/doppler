import { getDevice } from '../../device.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

export async function runAdam(
  params,
  grads,
  moment1,
  moment2,
  options = {}
) {
  const device = getDevice();
  const { count, step = 1, lr, beta1, beta2, eps } = options;

  const bytesPerElement = dtypeBytes(params.dtype);
  const inferredCount = count ?? Math.floor(params.buffer.size / bytesPerElement);
  const pipeline = await createPipeline('adam', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'adam_uniforms',
    32,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setUint32(4, step, true);
      view.setFloat32(8, lr, true);
      view.setFloat32(12, beta1, true);
      view.setFloat32(16, beta2, true);
      view.setFloat32(20, eps, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'adam_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.buffer } },
      { binding: 2, resource: { buffer: grads.buffer } },
      { binding: 3, resource: { buffer: moment1.buffer } },
      { binding: 4, resource: { buffer: moment2.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'adam');

  uniformBuffer.destroy();

  return createTensor(params.buffer, params.dtype, [...params.shape], 'adam_params');
}

export async function recordAdam(
  recorder,
  params,
  grads,
  moment1,
  moment2,
  options = {}
) {
  const device = recorder.device;
  const { count, step = 1, lr, beta1, beta2, eps } = options;

  const bytesPerElement = dtypeBytes(params.dtype);
  const inferredCount = count ?? Math.floor(params.buffer.size / bytesPerElement);
  const pipeline = await createPipeline('adam', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'adam_uniforms',
    32,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setUint32(4, step, true);
      view.setFloat32(8, lr, true);
      view.setFloat32(12, beta1, true);
      view.setFloat32(16, beta2, true);
      view.setFloat32(20, eps, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'adam_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.buffer } },
      { binding: 2, resource: { buffer: grads.buffer } },
      { binding: 3, resource: { buffer: moment1.buffer } },
      { binding: 4, resource: { buffer: moment2.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'adam');

  return createTensor(params.buffer, params.dtype, [...params.shape], 'adam_params');
}
