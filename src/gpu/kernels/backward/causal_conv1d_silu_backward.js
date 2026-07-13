import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

function validate(input, weight, gradOutput, options) {
  const numTokens = Math.floor(Number(options?.numTokens));
  const channels = Math.floor(Number(options?.channels));
  const kernelSize = Math.floor(Number(options?.kernelSize));
  if (numTokens < 1 || channels < 1 || kernelSize < 1) {
    throw new Error('causal conv1d SiLU backward requires numTokens, channels, and kernelSize.');
  }
  for (const [label, tensor] of [['input', input], ['weight', weight], ['gradOutput', gradOutput]]) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`causal conv1d SiLU backward requires f32 ${label}.`);
    }
  }
  return { numTokens, channels, kernelSize };
}

async function execute(recorder, input, weight, gradOutput, options = {}) {
  const dims = validate(input, weight, gradOutput, options);
  const device = recorder?.device || getDevice();
  if (!device) throw new Error('causal conv1d SiLU backward requires an active GPU device.');
  const outputBuffer = options.outputBuffer
    || acquireBuffer(dims.numTokens * dims.channels * 4, undefined, 'causal_conv1d_silu_backward_output');
  const pipeline = await createPipeline('causal_conv1d_silu_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'causal_conv1d_silu_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, dims.numTokens, true);
      view.setUint32(4, dims.channels, true);
      view.setUint32(8, dims.kernelSize, true);
    },
    recorder || null,
    device
  );
  const bindGroup = device.createBindGroup({
    label: 'causal_conv1d_silu_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: gradOutput.buffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
    ],
  });
  const workgroups = Math.ceil((dims.numTokens * dims.channels) / WORKGROUP_SIZES.DEFAULT);
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, 'causal_conv1d_silu_backward');
  } else {
    dispatch(device, pipeline, bindGroup, workgroups, 'causal_conv1d_silu_backward');
    releaseUniformBuffer(uniformBuffer);
  }
  return createTensor(
    outputBuffer,
    'f32',
    [dims.numTokens, dims.channels],
    'causal_conv1d_silu_backward_output'
  );
}

export function runCausalConv1dSiluBackward(input, weight, gradOutput, options = {}) {
  return execute(null, input, weight, gradOutput, options);
}

export function recordCausalConv1dSiluBackward(recorder, input, weight, gradOutput, options = {}) {
  return execute(recorder, input, weight, gradOutput, options);
}
