import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

function resolveOptions(gradOutput, options) {
  const numTokens = options.numTokens ?? gradOutput.shape?.[0] ?? null;
  const hiddenSize = options.hiddenSize ?? gradOutput.shape?.[1] ?? null;
  const vocabSize = options.vocabSize ?? null;
  if (!numTokens || !hiddenSize || !vocabSize) {
    throw new Error('embed backward requires numTokens, hiddenSize, and vocabSize');
  }
  const transpose = options.transpose === true;
  const indexOffset = options.indexOffset ?? 0;
  if (!Number.isFinite(indexOffset) || indexOffset < 0) {
    throw new Error('embed backward requires a non-negative indexOffset');
  }
  return { numTokens, hiddenSize, vocabSize, transpose, indexOffset };
}

export async function runEmbedBackward(indices, gradOutput, options = {}) {
  const device = getDevice();
  if (gradOutput.dtype !== 'f32') {
    throw new Error('embed backward requires f32 gradOutput');
  }

  const { numTokens, hiddenSize, vocabSize, transpose, indexOffset } = resolveOptions(gradOutput, options);
  const { outputBuffer = null } = options;
  const outputSize = vocabSize * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'embed_backward_output');

  const pipeline = await createPipeline('embed_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'embed_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
      view.setUint32(16, indexOffset, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'embed_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices.buffer } },
      { binding: 2, resource: { buffer: gradOutput.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: 'embed_backward_encoder' });
  encoder.clearBuffer(outputBuf);
  const pass = encoder.beginComputePass({ label: 'embed_backward_pass' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();

  const shape = transpose ? [hiddenSize, vocabSize] : [vocabSize, hiddenSize];
  return createTensor(outputBuf, 'f32', shape, 'embed_backward_output');
}

export async function recordEmbedBackward(recorder, indices, gradOutput, options = {}) {
  if (gradOutput.dtype !== 'f32') {
    throw new Error('embed backward requires f32 gradOutput');
  }

  const { numTokens, hiddenSize, vocabSize, transpose, indexOffset } = resolveOptions(gradOutput, options);
  const { outputBuffer = null } = options;
  const outputSize = vocabSize * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'embed_backward_output');

  const pipeline = await createPipeline('embed_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'embed_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
      view.setUint32(16, indexOffset, true);
    },
    recorder
  );

  const bindGroup = recorder.device.createBindGroup({
    label: 'embed_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices.buffer } },
      { binding: 2, resource: { buffer: gradOutput.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  recorder.getEncoder().clearBuffer(outputBuf);
  const pass = recorder.beginComputePass('embed_backward');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT));
  pass.end();

  const shape = transpose ? [hiddenSize, vocabSize] : [vocabSize, hiddenSize];
  return createTensor(outputBuf, 'f32', shape, 'embed_backward_output');
}
