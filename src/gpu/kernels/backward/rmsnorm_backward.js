import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { castF16ToF32, recordCastF16ToF32 } from '../cast.js';

export async function runRmsNormBackward(input, weight, gradOutput, options = {}) {
  const device = getDevice();
  const { numTokens, hiddenSize, eps = 1e-6, outputBuffer = null } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('rmsnorm backward requires numTokens and hiddenSize');
  }

  const inputTensor = input.dtype === 'f16' ? await castF16ToF32(input) : input;
  const weightTensor = weight.dtype === 'f16' ? await castF16ToF32(weight) : weight;
  const gradTensor = gradOutput.dtype === 'f16' ? await castF16ToF32(gradOutput) : gradOutput;

  const outputSize = numTokens * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_backward_output');

  const pipeline = await createPipeline('rmsnorm_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'rmsnorm_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setFloat32(8, eps, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'rmsnorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: weightTensor.buffer } },
      { binding: 3, resource: { buffer: gradTensor.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numTokens;
  dispatch(device, pipeline, bindGroup, workgroups, 'rmsnorm_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'rmsnorm_backward_output');
}

export async function recordRmsNormBackward(recorder, input, weight, gradOutput, options = {}) {
  const device = recorder.device;
  const { numTokens, hiddenSize, eps = 1e-6, outputBuffer = null } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('rmsnorm backward requires numTokens and hiddenSize');
  }

  const inputTensor = input.dtype === 'f16' ? await recordCastF16ToF32(recorder, input) : input;
  const weightTensor = weight.dtype === 'f16' ? await recordCastF16ToF32(recorder, weight) : weight;
  const gradTensor = gradOutput.dtype === 'f16' ? await recordCastF16ToF32(recorder, gradOutput) : gradOutput;

  const outputSize = numTokens * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_backward_output');

  const pipeline = await createPipeline('rmsnorm_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'rmsnorm_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setFloat32(8, eps, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'rmsnorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputTensor.buffer } },
      { binding: 2, resource: { buffer: weightTensor.buffer } },
      { binding: 3, resource: { buffer: gradTensor.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = numTokens;
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'rmsnorm_backward');

  return createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'rmsnorm_backward_output');
}
