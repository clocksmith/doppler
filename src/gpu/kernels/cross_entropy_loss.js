
import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';

function resolveDimensions(softmax, options) {
  const inferred = softmax.shape.length >= 2 ? softmax.shape : [];
  const numTokens = options.numTokens ?? inferred[0];
  const vocabSize = options.vocabSize ?? inferred[1];
  if (!numTokens || !vocabSize) {
    throw new Error('cross entropy loss requires numTokens and vocabSize');
  }
  return { numTokens, vocabSize };
}

export async function runCrossEntropyLoss(softmax, targets, options = {}) {
  const device = getDevice();
  const { outputBuffer = null } = options;
  const { numTokens, vocabSize } = resolveDimensions(softmax, options);

  const inputTensor = softmax.dtype === 'f16' ? await castF16ToF32(softmax) : softmax;
  const outputSize = numTokens * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'cross_entropy_loss_output');

  const pipeline = await createPipeline('cross_entropy_loss', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'cross_entropy_loss_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, vocabSize, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'cross_entropy_loss_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: targets.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(numTokens / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'cross_entropy_loss');

  uniformBuffer.destroy();

  if (inputTensor !== softmax) {
    releaseBuffer(inputTensor.buffer);
  }

  return createTensor(outputBuf, 'f32', [numTokens], 'cross_entropy_loss_output');
}

export async function recordCrossEntropyLoss(recorder, softmax, targets, options = {}) {
  const device = recorder.device;
  const { outputBuffer = null } = options;
  const { numTokens, vocabSize } = resolveDimensions(softmax, options);

  const inputTensor = softmax.dtype === 'f16' ? await recordCastF16ToF32(recorder, softmax) : softmax;
  if (inputTensor !== softmax) {
    recorder.trackTemporaryBuffer(inputTensor.buffer);
  }

  const outputSize = numTokens * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'cross_entropy_loss_output');

  const pipeline = await createPipeline('cross_entropy_loss', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'cross_entropy_loss_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, vocabSize, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'cross_entropy_loss_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: targets.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(numTokens / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'cross_entropy_loss');

  return createTensor(outputBuf, 'f32', [numTokens], 'cross_entropy_loss_output');
}
