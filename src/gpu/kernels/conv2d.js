
import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';


function selectConv2DVariant(isF16) {
  return selectRuleValue('conv2d', 'variant', { isF16 });
}


export async function runConv2D(
  input,
  weight,
  bias,
  options = {}
) {
  const device = getDevice();
  const {
    inChannels,
    outChannels,
    height,
    width,
    kernelH,
    kernelW,
    stride = 1,
    pad = 0,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(inChannels) || !Number.isFinite(outChannels) ||
      !Number.isFinite(height) || !Number.isFinite(width) ||
      !Number.isFinite(kernelH) || !Number.isFinite(kernelW)) {
    throw new Error('Conv2D requires explicit dimensions.');
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectConv2DVariant(isF16);
  const pipeline = await createPipeline('conv2d', variant);

  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  if (outHeight <= 0 || outWidth <= 0) {
    throw new Error(`Conv2D invalid output size: ${outHeight}x${outWidth}`);
  }

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'conv2d_output');

  const uniformBuffer = createUniformBufferWithView(
    'conv2d_uniforms',
    48,
    (view) => {
      view.setUint32(0, inChannels, true);
      view.setUint32(4, outChannels, true);
      view.setUint32(8, height, true);
      view.setUint32(12, width, true);
      view.setUint32(16, outHeight, true);
      view.setUint32(20, outWidth, true);
      view.setUint32(24, kernelH, true);
      view.setUint32(28, kernelW, true);
      view.setUint32(32, stride, true);
      view.setUint32(36, pad, true);
      view.setUint32(40, 0, true);
      view.setUint32(44, 0, true);
    },
    null,
    device
  );

  const weightBuffer = getBuffer(weight);
  let biasBuffer = getBuffer(bias);
  let tempBias = null;
  if (!biasBuffer) {
    const biasSize = outChannels * bytesPerElement;
    tempBias = acquireBuffer(biasSize, undefined, 'conv2d_bias_zero');
    biasBuffer = tempBias;
    device.queue.writeBuffer(biasBuffer, 0, new Uint8Array(biasSize));
  }

  const bindGroup = device.createBindGroup({
    label: 'conv2d_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: biasBuffer } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((outChannels * outHeight * outWidth) / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'conv2d');

  uniformBuffer.destroy();
  if (tempBias) {
    releaseBuffer(tempBias);
  }

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'conv2d_output');
}


export async function recordConv2D(
  recorder,
  input,
  weight,
  bias,
  options = {}
) {
  const device = recorder.device;
  const {
    inChannels,
    outChannels,
    height,
    width,
    kernelH,
    kernelW,
    stride = 1,
    pad = 0,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(inChannels) || !Number.isFinite(outChannels) ||
      !Number.isFinite(height) || !Number.isFinite(width) ||
      !Number.isFinite(kernelH) || !Number.isFinite(kernelW)) {
    throw new Error('Conv2D requires explicit dimensions.');
  }

  const isF16 = input.dtype === 'f16';
  const variant = selectConv2DVariant(isF16);
  const pipeline = await createPipeline('conv2d', variant);

  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  if (outHeight <= 0 || outWidth <= 0) {
    throw new Error(`Conv2D invalid output size: ${outHeight}x${outWidth}`);
  }

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'conv2d_output');

  const uniformBuffer = createUniformBufferWithView(
    'conv2d_uniforms',
    48,
    (view) => {
      view.setUint32(0, inChannels, true);
      view.setUint32(4, outChannels, true);
      view.setUint32(8, height, true);
      view.setUint32(12, width, true);
      view.setUint32(16, outHeight, true);
      view.setUint32(20, outWidth, true);
      view.setUint32(24, kernelH, true);
      view.setUint32(28, kernelW, true);
      view.setUint32(32, stride, true);
      view.setUint32(36, pad, true);
      view.setUint32(40, 0, true);
      view.setUint32(44, 0, true);
    },
    recorder
  );

  const weightBuffer = getBuffer(weight);
  let biasBuffer = getBuffer(bias);
  let tempBias = null;
  if (!biasBuffer) {
    const biasSize = outChannels * bytesPerElement;
    tempBias = acquireBuffer(biasSize, undefined, 'conv2d_bias_zero');
    biasBuffer = tempBias;
    device.queue.writeBuffer(biasBuffer, 0, new Uint8Array(biasSize));
  }

  const bindGroup = device.createBindGroup({
    label: 'conv2d_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: biasBuffer } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((outChannels * outHeight * outWidth) / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'conv2d');
  if (tempBias) {
    recorder.trackTemporaryBuffer(tempBias);
  }

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'conv2d_output');
}
