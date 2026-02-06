import { getDevice } from '../device.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';

function resolveCount(tensor, countOverride) {
  if (Number.isFinite(countOverride) && countOverride > 0) {
    return Math.floor(countOverride);
  }
  if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
    return tensor.shape.reduce((acc, value) => acc * value, 1);
  }
  return Math.floor(tensor.buffer.size / dtypeBytes(tensor.dtype));
}

export async function runClamp(
  input,
  minValue,
  maxValue,
  options = {}
) {
  const device = getDevice();
  if (input.dtype !== 'f32') {
    throw new Error(`runClamp: unsupported dtype ${input.dtype}.`);
  }

  const { count } = options;
  const inferredCount = resolveCount(input, count);
  const pipeline = await createPipeline('clamp', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'clamp_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(8, minValue, true);
      view.setFloat32(12, maxValue, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'clamp_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'clamp');

  uniformBuffer.destroy();

  return createTensor(input.buffer, input.dtype, [...input.shape], 'clamp_output');
}

export async function recordClamp(
  recorder,
  input,
  minValue,
  maxValue,
  options = {}
) {
  const device = recorder.device;
  if (input.dtype !== 'f32') {
    throw new Error(`recordClamp: unsupported dtype ${input.dtype}.`);
  }

  const { count } = options;
  const inferredCount = resolveCount(input, count);
  const pipeline = await createPipeline('clamp', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'clamp_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(8, minValue, true);
      view.setFloat32(12, maxValue, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'clamp_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'clamp');

  return createTensor(input.buffer, input.dtype, [...input.shape], 'clamp_output');
}

