import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';

function validate(input, weight, options) {
  const numTokens = Math.floor(Number(options?.numTokens));
  const channels = Math.floor(Number(options?.channels));
  const kernelSize = Math.floor(Number(options?.kernelSize));
  if (numTokens < 1 || channels < 1 || kernelSize < 1) {
    throw new Error('causal Conv1D SiLU requires numTokens, channels, and kernelSize.');
  }
  if (input?.dtype !== 'f32' || weight?.dtype !== 'f32') {
    throw new Error('causal Conv1D SiLU requires f32 input and weight.');
  }
  return { numTokens, channels, kernelSize };
}

async function execute(recorder, input, weight, options = {}) {
  const dims = validate(input, weight, options);
  const device = recorder?.device || getDevice();
  if (!device) throw new Error('causal Conv1D SiLU requires an active GPU device.');
  const ownsOutput = options.outputBuffer == null;
  const outputBuffer = options.outputBuffer
    || acquireBuffer(dims.numTokens * dims.channels * 4, undefined, 'causal_conv1d_silu_output');
  let uniformBuffer = null;
  let completed = false;
  try {
    const pipeline = await createPipeline('causal_conv1d_silu', 'default');
    uniformBuffer = createUniformBufferWithView(
      'causal_conv1d_silu_uniforms',
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
      label: 'causal_conv1d_silu_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: weight.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });
    const workgroups = Math.ceil((dims.numTokens * dims.channels) / WORKGROUP_SIZES.DEFAULT);
    if (recorder) {
      recordDispatch(recorder, pipeline, bindGroup, workgroups, 'causal_conv1d_silu');
    } else {
      dispatch(device, pipeline, bindGroup, workgroups, 'causal_conv1d_silu');
    }
    completed = true;
    return createTensor(
      outputBuffer,
      'f32',
      [dims.numTokens, dims.channels],
      'causal_conv1d_silu_output'
    );
  } finally {
    if (!recorder && uniformBuffer) releaseUniformBuffer(uniformBuffer);
    if (!completed && ownsOutput) releaseBuffer(outputBuffer);
  }
}

export function runCausalConv1dSilu(input, weight, options = {}) {
  return execute(null, input, weight, options);
}

export function recordCausalConv1dSilu(recorder, input, weight, options = {}) {
  return execute(recorder, input, weight, options);
}
