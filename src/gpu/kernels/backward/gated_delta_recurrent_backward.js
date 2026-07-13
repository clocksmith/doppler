import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { dispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

function validate(inputs, options) {
  const numTokens = Math.floor(Number(options?.numTokens));
  const numHeads = Math.floor(Number(options?.numHeads));
  const keyDim = Math.floor(Number(options?.keyDim));
  const valueDim = Math.floor(Number(options?.valueDim));
  const queryScale = Number(options?.queryScale);
  if (numTokens < 1 || numHeads < 1 || keyDim < 1 || valueDim < 1 || valueDim > 128) {
    throw new Error('gated-delta recurrent backward requires positive dimensions and valueDim <= 128.');
  }
  if (!Number.isFinite(queryScale)) {
    throw new Error('gated-delta recurrent backward requires finite queryScale.');
  }
  for (const [label, tensor] of Object.entries(inputs)) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`gated-delta recurrent backward requires f32 ${label}.`);
    }
  }
  return { numTokens, numHeads, keyDim, valueDim, queryScale };
}

function allocate(bytes, label, provided) {
  return provided || acquireBuffer(bytes, undefined, label);
}

export async function runGatedDeltaRecurrentBackward(inputs, options = {}) {
  const dims = validate(inputs, options);
  const device = getDevice();
  if (!device) throw new Error('gated-delta recurrent backward requires an active GPU device.');
  const queryBytes = dims.numTokens * dims.numHeads * dims.keyDim * 4;
  const valueBytes = dims.numTokens * dims.numHeads * dims.valueDim * 4;
  const scalarBytes = dims.numTokens * dims.numHeads * 4;
  const stateBytes = dims.numHeads * dims.keyDim * dims.valueDim * 4;
  const gradQueryBuffer = allocate(queryBytes, 'gated_delta_grad_query', options.gradQueryBuffer);
  const gradKeyBuffer = allocate(queryBytes, 'gated_delta_grad_key', options.gradKeyBuffer);
  const gradValueBuffer = allocate(valueBytes, 'gated_delta_grad_value', options.gradValueBuffer);
  const gradLogDecayBuffer = allocate(scalarBytes, 'gated_delta_grad_log_decay', options.gradLogDecayBuffer);
  const gradBetaBuffer = allocate(scalarBytes, 'gated_delta_grad_beta', options.gradBetaBuffer);
  const gradStateBuffer = allocate(stateBytes, 'gated_delta_grad_initial_state', options.gradStateBuffer);
  device.queue.writeBuffer(gradStateBuffer, 0, new Uint8Array(stateBytes));

  const pipeline = await createPipeline('gated_delta_recurrent_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'gated_delta_recurrent_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, dims.numTokens, true);
      view.setUint32(4, dims.numHeads, true);
      view.setUint32(8, dims.keyDim, true);
      view.setUint32(12, dims.valueDim, true);
      view.setFloat32(16, dims.queryScale, true);
    },
    null,
    device
  );
  const bindGroup = device.createBindGroup({
    label: 'gated_delta_recurrent_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputs.query.buffer } },
      { binding: 2, resource: { buffer: inputs.key.buffer } },
      { binding: 3, resource: { buffer: inputs.value.buffer } },
      { binding: 4, resource: { buffer: inputs.logDecay.buffer } },
      { binding: 5, resource: { buffer: inputs.beta.buffer } },
      { binding: 6, resource: { buffer: inputs.stateHistory.buffer } },
      { binding: 7, resource: { buffer: inputs.gradOutput.buffer } },
      { binding: 8, resource: { buffer: gradQueryBuffer } },
      { binding: 9, resource: { buffer: gradKeyBuffer } },
      { binding: 10, resource: { buffer: gradValueBuffer } },
      { binding: 11, resource: { buffer: gradLogDecayBuffer } },
      { binding: 12, resource: { buffer: gradBetaBuffer } },
      { binding: 13, resource: { buffer: gradStateBuffer } },
    ],
  });
  dispatch(device, pipeline, bindGroup, dims.numHeads, 'gated_delta_recurrent_backward');
  releaseUniformBuffer(uniformBuffer);
  return {
    query: createTensor(gradQueryBuffer, 'f32', [dims.numTokens, dims.numHeads, dims.keyDim], 'gated_delta_grad_query'),
    key: createTensor(gradKeyBuffer, 'f32', [dims.numTokens, dims.numHeads, dims.keyDim], 'gated_delta_grad_key'),
    value: createTensor(gradValueBuffer, 'f32', [dims.numTokens, dims.numHeads, dims.valueDim], 'gated_delta_grad_value'),
    logDecay: createTensor(gradLogDecayBuffer, 'f32', [dims.numTokens, dims.numHeads], 'gated_delta_grad_log_decay'),
    beta: createTensor(gradBetaBuffer, 'f32', [dims.numTokens, dims.numHeads], 'gated_delta_grad_beta'),
    initialState: createTensor(gradStateBuffer, 'f32', [dims.numHeads, dims.keyDim, dims.valueDim], 'gated_delta_grad_initial_state'),
  };
}
