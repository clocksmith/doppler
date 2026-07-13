import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

export async function runQwenAttentionSplitQGate(input, options = {}) {
  const numTokens = positiveInteger(options.numTokens, 'numTokens');
  const numHeads = positiveInteger(options.numHeads, 'numHeads');
  const headDim = positiveInteger(options.headDim, 'headDim');
  const elementCount = numTokens * numHeads * headDim;
  if (input?.dtype !== 'f32'
    || input.shape?.reduce((product, value) => product * value, 1) !== elementCount * 2) {
    throw new Error('Qwen attention Q/gate split requires correctly shaped f32 input.');
  }
  const device = getDevice();
  if (!device) throw new Error('Qwen attention Q/gate split requires an active GPU device.');
  const bytes = elementCount * Float32Array.BYTES_PER_ELEMENT;
  const queryBuffer = acquireBuffer(bytes, undefined, 'qwen_attention_query');
  const gateBuffer = acquireBuffer(bytes, undefined, 'qwen_attention_gate');
  let uniformBuffer = null;
  let completed = false;
  try {
    const pipeline = await createPipeline('qwen_attention_split_q_gate', 'default');
    uniformBuffer = createUniformBufferWithView(
      'qwen_attention_split_q_gate_uniforms',
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
      label: 'qwen_attention_split_q_gate_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: queryBuffer } },
        { binding: 3, resource: { buffer: gateBuffer } },
      ],
    });
    dispatch(
      device,
      pipeline,
      bindGroup,
      Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT),
      'qwen_attention_split_q_gate'
    );
    completed = true;
    return {
      query: createTensor(
        queryBuffer,
        'f32',
        [numTokens, numHeads, headDim],
        'qwen_attention_query'
      ),
      gate: createTensor(
        gateBuffer,
        'f32',
        [numTokens, numHeads, headDim],
        'qwen_attention_gate'
      ),
    };
  } finally {
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    if (!completed) {
      releaseBuffer(queryBuffer);
      releaseBuffer(gateBuffer);
    }
  }
}
