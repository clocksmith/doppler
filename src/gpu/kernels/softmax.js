

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';


const SOFTMAX_SMALL_THRESHOLD = 256;


function selectSoftmaxVariant(innerSize) {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;

  if (hasSubgroups) {
    if (innerSize <= SOFTMAX_SMALL_THRESHOLD) {
      return 'small_subgroup';
    }
    return 'subgroup';
  }

  if (innerSize <= SOFTMAX_SMALL_THRESHOLD) {
    return 'small';
  }
  return 'default';
}


export async function runSoftmax(
  input,
  axis,
  options = {}
) {
  const device = getDevice();
  const { batchSize = 1, size, temperature = 1.0, outputBuffer = null } = options;

  const bytesPerElement = input.dtype === 'f16' ? 2 : 4;
  const inferredSize = size || (input.buffer.size / (batchSize * bytesPerElement));
  const variant = selectSoftmaxVariant(inferredSize);
  trace.kernels(`Softmax: size=${inferredSize}, variant=${variant}`);
  const pipeline = await createPipeline('softmax', variant);

  const outputSize = batchSize * inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'softmax_output');

  // Create uniform buffer
  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
  const uniformBuffer = createUniformBufferWithView(
    'softmax_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);  // innerSize at offset 0
      view.setUint32(4, batchSize, true);     // outerSize at offset 4
      view.setFloat32(8, temperature, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  dispatch(device, pipeline, bindGroup, batchSize, 'softmax');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [batchSize, inferredSize], 'softmax_output');
}


export async function runSoftmaxTopK(
  logits,
  numTokens,
  numExperts,
  topK,
  options = {}
) {
  const device = getDevice();
  const { normalize = true } = options;

  const pipeline = await createPipeline('topk', 'fused');

  // Output buffers: indices [numTokens, topK] as u32, weights [numTokens, topK] as f32
  const indicesSize = numTokens * topK * 4; // u32
  const weightsSize = numTokens * topK * 4; // f32

  const indices = acquireBuffer(indicesSize, undefined, 'softmax_topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'softmax_topk_weights');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'softmax_topk_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  });

  dispatch(device, pipeline, bindGroup, numTokens, 'softmax_topk');

  uniformBuffer.destroy();

  return { indices, weights };
}


export async function recordSoftmax(
  recorder,
  input,
  axis,
  options = {}
) {
  const device = recorder.device;
  const {
    batchSize = 1,
    seqLen = null,
    outputBuffer = null,
  } = options;

  const bytesPerElement = input.dtype === 'f16' ? 2 : 4;
  const inferredSeqLen = seqLen || (input.buffer.size / (batchSize * bytesPerElement));
  const pipeline = await createPipeline('softmax', 'default');

  const outputSize = batchSize * inferredSeqLen * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'softmax_output');

  // Uniform buffer
  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
  const uniformBuffer = createUniformBufferWithView(
    'softmax_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSeqLen, true);  // innerSize at offset 0
      view.setUint32(4, batchSize, true);       // outerSize at offset 4
      view.setFloat32(8, 1.0, true);            // temperature (default 1.0)
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, batchSize, 'softmax');

  return createTensor(output, input.dtype, [batchSize, inferredSeqLen], 'softmax_output');
}
