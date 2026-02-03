import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectPixelShuffleVariant(dtype) {
  return selectRuleValue('pixel_shuffle', 'variant', { dtype });
}

export async function runPixelShuffle(input, options = {}) {
  const device = getDevice();
  const {
    outChannels,
    outHeight,
    outWidth,
    gridWidth,
    gridHeight,
    patchSize,
    patchChannels,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(outChannels) || !Number.isFinite(outHeight) || !Number.isFinite(outWidth) ||
      !Number.isFinite(gridWidth) || !Number.isFinite(gridHeight) || !Number.isFinite(patchSize)) {
    throw new Error('PixelShuffle requires explicit dimensions.');
  }

  const inferredPatchChannels = patchChannels ?? outChannels * patchSize * patchSize;
  const variant = selectPixelShuffleVariant(input.dtype);
  const pipeline = await createPipeline('pixel_shuffle', variant);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'pixel_shuffle_output');

  const uniformBuffer = createUniformBufferWithView(
    'pixel_shuffle_uniforms',
    32,
    (view) => {
      view.setUint32(0, outChannels, true);
      view.setUint32(4, outHeight, true);
      view.setUint32(8, outWidth, true);
      view.setUint32(12, gridWidth, true);
      view.setUint32(16, gridHeight, true);
      view.setUint32(20, patchSize, true);
      view.setUint32(24, inferredPatchChannels, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'pixel_shuffle_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((outChannels * outHeight * outWidth) / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'pixel_shuffle');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'pixel_shuffle_output');
}

export async function recordPixelShuffle(recorder, input, options = {}) {
  const device = recorder.device;
  const {
    outChannels,
    outHeight,
    outWidth,
    gridWidth,
    gridHeight,
    patchSize,
    patchChannels,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(outChannels) || !Number.isFinite(outHeight) || !Number.isFinite(outWidth) ||
      !Number.isFinite(gridWidth) || !Number.isFinite(gridHeight) || !Number.isFinite(patchSize)) {
    throw new Error('PixelShuffle requires explicit dimensions.');
  }

  const inferredPatchChannels = patchChannels ?? outChannels * patchSize * patchSize;
  const variant = selectPixelShuffleVariant(input.dtype);
  const pipeline = await createPipeline('pixel_shuffle', variant);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'pixel_shuffle_output');

  const uniformBuffer = createUniformBufferWithView(
    'pixel_shuffle_uniforms',
    32,
    (view) => {
      view.setUint32(0, outChannels, true);
      view.setUint32(4, outHeight, true);
      view.setUint32(8, outWidth, true);
      view.setUint32(12, gridWidth, true);
      view.setUint32(16, gridHeight, true);
      view.setUint32(20, patchSize, true);
      view.setUint32(24, inferredPatchChannels, true);
      view.setUint32(28, 0, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'pixel_shuffle_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((outChannels * outHeight * outWidth) / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'pixel_shuffle');

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'pixel_shuffle_output');
}
