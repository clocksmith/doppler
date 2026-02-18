import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { getDevice } from '../../device.js';

export async function runLayerNormBackward(input, weight, gradOutput, options = {}) {
  const {
    numTokens,
    hiddenSize,
    eps = 1e-5,
    outputBuffer = null,
    gradWeightBuffer = null,
    gradBiasBuffer = null,
  } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('layernorm backward requires numTokens and hiddenSize');
  }

  const device = getDevice();
  const outputSize = numTokens * hiddenSize * 4; // f32
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_backward_output');
  
  const weightGradSize = hiddenSize * 4;
  const weightGradBuf = gradWeightBuffer || acquireBuffer(weightGradSize, undefined, 'layernorm_backward_weight_grad');
  const biasGradBuf = gradBiasBuffer || acquireBuffer(weightGradSize, undefined, 'layernorm_backward_bias_grad');

  // Zero out atomics
  device.queue.writeBuffer(weightGradBuf, 0, new Float32Array(hiddenSize));
  device.queue.writeBuffer(biasGradBuf, 0, new Float32Array(hiddenSize));

  const pipeline = await createPipeline('layernorm_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'layernorm_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, hiddenSize, true);
      view.setUint32(4, numTokens, true);
      view.setFloat32(8, eps, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'layernorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: weightGradBuf } },
      { binding: 6, resource: { buffer: biasGradBuf } },
    ],
  });

  const workgroups = numTokens; // 1 workgroup per token
  dispatch(device, pipeline, bindGroup, workgroups, 'layernorm_backward');

  uniformBuffer.destroy();

  return {
    gradInput: createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'layernorm_backward_output'),
    gradWeight: createTensor(weightGradBuf, 'f32', [hiddenSize], 'layernorm_backward_weight_grad'),
    gradBias: createTensor(biasGradBuf, 'f32', [hiddenSize], 'layernorm_backward_bias_grad'),
  };
}

export async function recordLayerNormBackward(recorder, input, weight, gradOutput, options = {}) {
  const {
    numTokens,
    hiddenSize,
    eps = 1e-5,
    outputBuffer = null,
    gradWeightBuffer = null,
    gradBiasBuffer = null,
  } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('layernorm backward requires numTokens and hiddenSize');
  }

  const device = recorder.device;
  const outputSize = numTokens * hiddenSize * 4; // f32
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_backward_output');
  
  const weightGradSize = hiddenSize * 4;
  const weightGradBuf = gradWeightBuffer || acquireBuffer(weightGradSize, undefined, 'layernorm_backward_weight_grad');
  const biasGradBuf = gradBiasBuffer || acquireBuffer(weightGradSize, undefined, 'layernorm_backward_bias_grad');

  // record zero out (using clearBuffer or similar if available, otherwise writeBuffer)
  // Note: CommandRecorder might not have clearBuffer yet, so we use device.queue for now if not recording a pass.
  // Ideally recorder.clearBuffer(weightGradBuf).
  device.queue.writeBuffer(weightGradBuf, 0, new Float32Array(hiddenSize));
  device.queue.writeBuffer(biasGradBuf, 0, new Float32Array(hiddenSize));

  const pipeline = await createPipeline('layernorm_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'layernorm_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, hiddenSize, true);
      view.setUint32(4, numTokens, true);
      view.setFloat32(8, eps, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'layernorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: weightGradBuf } },
      { binding: 6, resource: { buffer: biasGradBuf } },
    ],
  });

  const workgroups = numTokens;
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'layernorm_backward');

  return {
    gradInput: createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'layernorm_backward_output'),
    gradWeight: createTensor(weightGradBuf, 'f32', [hiddenSize], 'layernorm_backward_weight_grad'),
    gradBias: createTensor(biasGradBuf, 'f32', [hiddenSize], 'layernorm_backward_bias_grad'),
  };
}
