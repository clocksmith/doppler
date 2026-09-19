import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

async function runGatedBackward(input, gate, gradOutput, options = {}) {
  const count = Math.floor(Number(options.count));
  if (!Number.isInteger(count) || count < 1) {
    throw new Error('sigmoid-gated backward requires a positive count.');
  }
  for (const [label, tensor] of [['input', input], ['gate', gate], ['gradOutput', gradOutput]]) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`sigmoid-gated backward requires f32 ${label}.`);
    }
  }
  const device = getDevice();
  if (!device) throw new Error('sigmoid-gated backward requires an active GPU device.');
  const bytes = count * Float32Array.BYTES_PER_ELEMENT;
  const gradInputBuffer = acquireBuffer(bytes, undefined, 'sigmoid_gated_grad_input');
  const gradGateBuffer = acquireBuffer(bytes, undefined, 'sigmoid_gated_grad_gate');
  let uniformBuffer = null;
  let completed = false;
  try {
    const pipeline = await createPipeline('sigmoid_gated_backward', 'default');
    uniformBuffer = createUniformBufferWithView(
      'sigmoid_gated_backward_uniforms',
      16,
      (view) => {
        view.setUint32(0, count, true);
        view.setUint32(4, options.gateMode === 'silu' ? 1 : 0, true);
        view.setFloat32(8, Number.isFinite(options.swigluLimit) ? options.swigluLimit : 0, true);
      },
      null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'sigmoid_gated_backward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: gate.buffer } },
        { binding: 3, resource: { buffer: gradOutput.buffer } },
        { binding: 4, resource: { buffer: gradInputBuffer } },
        { binding: 5, resource: { buffer: gradGateBuffer } },
      ],
    });
    dispatch(
      device,
      pipeline,
      bindGroup,
      Math.ceil(count / WORKGROUP_SIZES.DEFAULT),
      'sigmoid_gated_backward'
    );
    completed = true;
    return {
      input: createTensor(gradInputBuffer, 'f32', [...input.shape], 'sigmoid_gated_grad_input'),
      gate: createTensor(gradGateBuffer, 'f32', [...gate.shape], 'sigmoid_gated_grad_gate'),
    };
  } finally {
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    if (!completed) {
      releaseBuffer(gradInputBuffer);
      releaseBuffer(gradGateBuffer);
    }
  }
}

export function runSigmoidGatedBackward(input, gate, gradOutput, options = {}) {
  return runGatedBackward(input, gate, gradOutput, { ...options, gateMode: 'sigmoid' });
}

export async function runSiluGatedBackward(gate, up, gradOutput, options = {}) {
  const gradients = await runGatedBackward(up, gate, gradOutput, {
    ...options,
    gateMode: 'silu',
  });
  return { gate: gradients.gate, up: gradients.input };
}
