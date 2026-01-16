

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';


export async function runSplitQKV(
  qkvTensor,
  options
) {
  const device = getDevice();
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;

  const outputDtype = qkvTensor.dtype;
  const pipelineVariant = selectRuleValue('splitQkv', 'variant', { outputDtype });
  const pipeline = await getPipelineFast('split_qkv', pipelineVariant);
  const bytesPerElement = dtypeBytes(outputDtype);

  // Allocate output buffers if not provided
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'split_qkv_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'split_qkv_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } },
    ],
  });

  // Dispatch - total elements across all outputs
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'split_qkv');

  uniformBuffer.destroy();

  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

  return { Q, K, V };
}


export async function recordSplitQKV(
  recorder,
  qkvTensor,
  options
) {
  const device = recorder.device;
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;

  const outputDtype = qkvTensor.dtype;
  const pipelineVariant = selectRuleValue('splitQkv', 'variant', { outputDtype });
  const pipeline = await getPipelineFast('split_qkv', pipelineVariant);
  const bytesPerElement = dtypeBytes(outputDtype);

  // Allocate output buffers if not provided
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'split_qkv_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    recorder
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'split_qkv_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } },
    ],
  });

  // Dispatch
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'split_qkv');

  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

  return { Q, K, V };
}
