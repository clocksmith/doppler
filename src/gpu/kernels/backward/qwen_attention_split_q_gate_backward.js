import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

export async function runQwenAttentionSplitQGateBackward(gradQuery, gradGate, options = {}) {
  const numTokens = positiveInteger(options.numTokens, 'numTokens');
  const numHeads = positiveInteger(options.numHeads, 'numHeads');
  const headDim = positiveInteger(options.headDim, 'headDim');
  const elementCount = numTokens * numHeads * headDim;
  for (const [label, tensor] of [['gradQuery', gradQuery], ['gradGate', gradGate]]) {
    if (tensor?.dtype !== 'f32'
      || tensor.shape?.reduce((product, value) => product * value, 1) !== elementCount) {
      throw new Error(`Qwen attention Q/gate split backward requires correctly shaped f32 ${label}.`);
    }
  }
  const device = getDevice();
  if (!device) throw new Error('Qwen attention Q/gate split backward requires an active GPU device.');
  const outputBuffer = acquireBuffer(
    elementCount * 2 * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    'qwen_attention_grad_q_gate'
  );
  let uniformBuffer = null;
  let completed = false;
  try {
    const pipeline = await createPipeline('qwen_attention_split_q_gate_backward', 'default');
    uniformBuffer = createUniformBufferWithView(
      'qwen_attention_split_q_gate_backward_uniforms',
      16,
      (view) => {
        view.setUint32(0, numTokens, true);
        view.setUint32(4, numHeads, true);
        view.setUint32(8, headDim, true);
        view.setUint32(12, elementCount, true);
      },
      null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'qwen_attention_split_q_gate_backward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gradQuery.buffer } },
        { binding: 2, resource: { buffer: gradGate.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });
    dispatch(
      device,
      pipeline,
      bindGroup,
      Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT),
      'qwen_attention_split_q_gate_backward'
    );
    completed = true;
    return createTensor(
      outputBuffer,
      'f32',
      [numTokens, numHeads, headDim * 2],
      'qwen_attention_grad_q_gate'
    );
  } finally {
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    if (!completed) releaseBuffer(outputBuffer);
  }
}
