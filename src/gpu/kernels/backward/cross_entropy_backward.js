import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

export async function runCrossEntropyBackward(softmax, targets, gradOutput, options = {}) {
  const device = getDevice();
  const { numTokens, vocabSize, outputBuffer = null } = options;

  if (!numTokens || !vocabSize) {
    throw new Error('cross entropy backward requires numTokens and vocabSize');
  }

  const bytesPerElement = dtypeBytes(softmax.dtype);
  const outputSize = numTokens * vocabSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'cross_entropy_backward_output');

  const pipeline = await createPipeline('cross_entropy_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'cross_entropy_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, vocabSize, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'cross_entropy_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: softmax.buffer } },
      { binding: 2, resource: { buffer: targets.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil((numTokens * vocabSize) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'cross_entropy_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, softmax.dtype, [numTokens, vocabSize], 'cross_entropy_backward_output');
}

export async function recordCrossEntropyBackward(recorder, softmax, targets, gradOutput, options = {}) {
  const device = recorder.device;
  const { numTokens, vocabSize, outputBuffer = null } = options;

  if (!numTokens || !vocabSize) {
    throw new Error('cross entropy backward requires numTokens and vocabSize');
  }

  const bytesPerElement = dtypeBytes(softmax.dtype);
  const outputSize = numTokens * vocabSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'cross_entropy_backward_output');

  const pipeline = await createPipeline('cross_entropy_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'cross_entropy_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, vocabSize, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'cross_entropy_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: softmax.buffer } },
      { binding: 2, resource: { buffer: targets.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil((numTokens * vocabSize) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'cross_entropy_backward');

  return createTensor(outputBuf, softmax.dtype, [numTokens, vocabSize], 'cross_entropy_backward_output');
}
