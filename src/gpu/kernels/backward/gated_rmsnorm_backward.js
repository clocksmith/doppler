import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

function validate(input, gate, weight, gradOutput, options) {
  const rows = Math.floor(Number(options?.rows));
  const width = Math.floor(Number(options?.width));
  const eps = Number(options?.eps);
  if (rows < 1 || width < 1 || !Number.isFinite(eps) || eps <= 0) {
    throw new Error('gated RMSNorm backward requires rows, width, and positive eps.');
  }
  for (const [label, tensor] of [
    ['input', input],
    ['gate', gate],
    ['weight', weight],
    ['gradOutput', gradOutput],
  ]) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`gated RMSNorm backward requires f32 ${label}.`);
    }
  }
  return { rows, width, eps };
}

async function execute(recorder, input, gate, weight, gradOutput, options = {}) {
  const dims = validate(input, gate, weight, gradOutput, options);
  const device = recorder?.device || getDevice();
  if (!device) throw new Error('gated RMSNorm backward requires an active GPU device.');
  const outputBytes = dims.rows * dims.width * 4;
  const gradInputBuffer = options.gradInputBuffer
    || acquireBuffer(outputBytes, undefined, 'gated_rmsnorm_backward_input');
  const gradGateBuffer = options.gradGateBuffer
    || acquireBuffer(outputBytes, undefined, 'gated_rmsnorm_backward_gate');
  const pipeline = await createPipeline('gated_rmsnorm_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'gated_rmsnorm_backward_uniforms',
    16,
    (view) => {
      view.setUint32(0, dims.rows, true);
      view.setUint32(4, dims.width, true);
      view.setFloat32(8, dims.eps, true);
    },
    recorder || null,
    device
  );
  const bindGroup = device.createBindGroup({
    label: 'gated_rmsnorm_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: gate.buffer } },
      { binding: 3, resource: { buffer: weight.buffer } },
      { binding: 4, resource: { buffer: gradOutput.buffer } },
      { binding: 5, resource: { buffer: gradInputBuffer } },
      { binding: 6, resource: { buffer: gradGateBuffer } },
    ],
  });
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, dims.rows, 'gated_rmsnorm_backward');
  } else {
    dispatch(device, pipeline, bindGroup, dims.rows, 'gated_rmsnorm_backward');
    releaseUniformBuffer(uniformBuffer);
  }
  return {
    gradInput: createTensor(
      gradInputBuffer,
      'f32',
      [dims.rows, dims.width],
      'gated_rmsnorm_backward_input'
    ),
    gradGate: createTensor(
      gradGateBuffer,
      'f32',
      [dims.rows, dims.width],
      'gated_rmsnorm_backward_gate'
    ),
  };
}

export function runGatedRmsNormBackward(input, gate, weight, gradOutput, options = {}) {
  return execute(null, input, gate, weight, gradOutput, options);
}

export function recordGatedRmsNormBackward(recorder, input, gate, weight, gradOutput, options = {}) {
  return execute(recorder, input, gate, weight, gradOutput, options);
}
