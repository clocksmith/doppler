import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch } from './dispatch.js';
import { createPipeline } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { requirePositiveInteger } from './kernel-geometry.js';
import { withKernelOutputs } from './kernel-execution.js';

const CONTEXT = 'Qwen attention Q/gate split';

export async function runQwenAttentionSplitQGate(input, options = {}) {
  const numTokens = requirePositiveInteger(options.numTokens, 'numTokens', CONTEXT);
  const numHeads = requirePositiveInteger(options.numHeads, 'numHeads', CONTEXT);
  const headDim = requirePositiveInteger(options.headDim, 'headDim', CONTEXT);
  const elementCount = numTokens * numHeads * headDim;
  if (input?.dtype !== 'f32'
    || input.shape?.reduce((product, value) => product * value, 1) !== elementCount * 2) {
    throw new Error('Qwen attention Q/gate split requires correctly shaped f32 input.');
  }
  const device = getDevice();
  if (!device) throw new Error('Qwen attention Q/gate split requires an active GPU device.');
  const bytes = elementCount * Float32Array.BYTES_PER_ELEMENT;
  // Both outputs are acquired as one transaction: a failed second acquisition
  // or dispatch rolls the first back, and success hands both to the tensors.
  return withKernelOutputs(device, {
    query: { bytes, label: 'qwen_attention_query' },
    gate: { bytes, label: 'qwen_attention_gate' },
  }, async ({ query, gate }) => {
    let uniformBuffer = null;
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
          { binding: 2, resource: { buffer: query } },
          { binding: 3, resource: { buffer: gate } },
        ],
      });
      dispatch(
        device,
        pipeline,
        bindGroup,
        Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT),
        'qwen_attention_split_q_gate'
      );
      return {
        query: createTensor(query, 'f32', [numTokens, numHeads, headDim], 'qwen_attention_query'),
        gate: createTensor(gate, 'f32', [numTokens, numHeads, headDim], 'qwen_attention_gate'),
      };
    } finally {
      if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    }
  });
}
