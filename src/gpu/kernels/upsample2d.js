import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectUpsample2DVariant(isF16) {
  return selectRuleValue('upsample2d', 'variant', { isF16 });
}

export async function runUpsample2D(
  input,
  options = {}
) {
  const device = getDevice();
  const {
    channels,
    height,
    width,
    scale = 2,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(channels) || !Number.isFinite(height) || !Number.isFinite(width)) {
    throw new Error('Upsample2D requires channels/height/width.');
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('Upsample2D requires scale > 0.');
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectUpsample2DVariant(isF16);
  const pipeline = await createPipeline('upsample2d', variant);

  const outHeight = height * scale;
  const outWidth = width * scale;
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'upsample2d_output');

  const uniformBuffer = createUniformBufferWithView(
    'upsample2d_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, outHeight, true);
      view.setUint32(16, outWidth, true);
      view.setUint32(20, scale, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'upsample2d_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((channels * outHeight * outWidth) / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'upsample2d');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [channels, outHeight, outWidth], 'upsample2d_output');
}

export async function recordUpsample2D(
  recorder,
  input,
  options = {}
) {
  const {
    channels,
    height,
    width,
    scale = 2,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(channels) || !Number.isFinite(height) || !Number.isFinite(width)) {
    throw new Error('Upsample2D requires channels/height/width.');
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('Upsample2D requires scale > 0.');
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectUpsample2DVariant(isF16);
  const pipeline = await createPipeline('upsample2d', variant);

  const outHeight = height * scale;
  const outWidth = width * scale;
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'upsample2d_output');

  const uniformBuffer = createUniformBufferWithView(
    'upsample2d_uniforms',
    32,
    (view) => {
      view.setUint32(0, channels, true);
      view.setUint32(4, height, true);
      view.setUint32(8, width, true);
      view.setUint32(12, outHeight, true);
      view.setUint32(16, outWidth, true);
      view.setUint32(20, scale, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'upsample2d_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((channels * outHeight * outWidth) / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'upsample2d');

  return createTensor(output, input.dtype, [channels, outHeight, outWidth], 'upsample2d_output');
}
