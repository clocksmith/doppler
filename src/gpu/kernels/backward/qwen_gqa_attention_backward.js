import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function requireElements(tensor, expected, label) {
  if (tensor?.dtype !== 'f32'
    || tensor.shape?.reduce((product, value) => product * value, 1) !== expected) {
    throw new Error(`Qwen GQA backward requires ${label} to be f32 with ${expected} elements.`);
  }
}

export async function runQwenGqaAttentionBackward(query, key, value, gradOutput, options = {}) {
  const seqLen = positiveInteger(options.seqLen, 'seqLen');
  const numHeads = positiveInteger(options.numHeads, 'numHeads');
  const numKVHeads = positiveInteger(options.numKVHeads, 'numKVHeads');
  const headDim = positiveInteger(options.headDim, 'headDim');
  const scale = Number(options.scale);
  if (numHeads % numKVHeads !== 0 || !Number.isFinite(scale)) {
    throw new Error('Qwen GQA backward requires divisible head counts and finite scale.');
  }
  const queryElements = seqLen * numHeads * headDim;
  const kvElements = seqLen * numKVHeads * headDim;
  const scoreElements = numHeads * seqLen * seqLen;
  requireElements(query, queryElements, 'query');
  requireElements(key, kvElements, 'key');
  requireElements(value, kvElements, 'value');
  requireElements(gradOutput, queryElements, 'gradOutput');
  const device = getDevice();
  if (!device) throw new Error('Qwen GQA backward requires an active GPU device.');
  const bytes = Float32Array.BYTES_PER_ELEMENT;
  const softmaxBuffer = acquireBuffer(scoreElements * bytes, undefined, 'qwen_gqa_softmax');
  const gradScoresBuffer = acquireBuffer(scoreElements * bytes, undefined, 'qwen_gqa_grad_scores');
  const gradQueryBuffer = acquireBuffer(queryElements * bytes, undefined, 'qwen_gqa_grad_query');
  const gradKeyBuffer = acquireBuffer(kvElements * bytes, undefined, 'qwen_gqa_grad_key');
  const gradValueBuffer = acquireBuffer(kvElements * bytes, undefined, 'qwen_gqa_grad_value');
  let uniformBuffer = null;
  let completed = false;
  try {
    const [softmaxPipeline, scoresPipeline, gradientsPipeline] = await Promise.all([
      createPipeline('qwen_gqa_softmax_recompute', 'default'),
      createPipeline('qwen_gqa_softmax_backward_scores', 'default'),
      createPipeline('qwen_gqa_attention_backward', 'default'),
    ]);
    uniformBuffer = createUniformBufferWithView(
      'qwen_gqa_attention_backward_uniforms',
      32,
      (view) => {
        view.setUint32(0, seqLen, true);
        view.setUint32(4, numHeads, true);
        view.setUint32(8, numKVHeads, true);
        view.setUint32(12, headDim, true);
        view.setFloat32(16, scale, true);
        view.setUint32(20, options.causal === false ? 0 : 1, true);
      },
      null,
      device
    );
    const softmaxBindGroup = device.createBindGroup({
      label: 'qwen_gqa_softmax_recompute_bind_group',
      layout: softmaxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: query.buffer } },
        { binding: 2, resource: { buffer: key.buffer } },
        { binding: 3, resource: { buffer: softmaxBuffer } },
      ],
    });
    const scoresBindGroup = device.createBindGroup({
      label: 'qwen_gqa_softmax_backward_scores_bind_group',
      layout: scoresPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: value.buffer } },
        { binding: 2, resource: { buffer: gradOutput.buffer } },
        { binding: 3, resource: { buffer: softmaxBuffer } },
        { binding: 4, resource: { buffer: gradScoresBuffer } },
      ],
    });
    const gradientsBindGroup = device.createBindGroup({
      label: 'qwen_gqa_attention_backward_bind_group',
      layout: gradientsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: query.buffer } },
        { binding: 2, resource: { buffer: key.buffer } },
        { binding: 3, resource: { buffer: softmaxBuffer } },
        { binding: 4, resource: { buffer: gradScoresBuffer } },
        { binding: 5, resource: { buffer: gradOutput.buffer } },
        { binding: 6, resource: { buffer: gradQueryBuffer } },
        { binding: 7, resource: { buffer: gradKeyBuffer } },
        { binding: 8, resource: { buffer: gradValueBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: 'qwen_gqa_attention_backward' });
    for (const [label, pipeline, bindGroup, workgroups] of [
      ['qwen_gqa_softmax_recompute', softmaxPipeline, softmaxBindGroup, seqLen * numHeads],
      ['qwen_gqa_softmax_backward_scores', scoresPipeline, scoresBindGroup, seqLen * numHeads],
      [
        'qwen_gqa_attention_backward_gradients',
        gradientsPipeline,
        gradientsBindGroup,
        Math.ceil((queryElements + (kvElements * 2)) / WORKGROUP_SIZES.DEFAULT),
      ],
    ]) {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    completed = true;
    return {
      query: createTensor(
        gradQueryBuffer,
        'f32',
        [seqLen, numHeads, headDim],
        'qwen_gqa_grad_query'
      ),
      key: createTensor(
        gradKeyBuffer,
        'f32',
        [seqLen, numKVHeads, headDim],
        'qwen_gqa_grad_key'
      ),
      value: createTensor(
        gradValueBuffer,
        'f32',
        [seqLen, numKVHeads, headDim],
        'qwen_gqa_grad_value'
      ),
    };
  } finally {
    await device.queue.onSubmittedWorkDone().catch(() => {});
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    releaseBuffer(softmaxBuffer);
    releaseBuffer(gradScoresBuffer);
    if (!completed) {
      releaseBuffer(gradQueryBuffer);
      releaseBuffer(gradKeyBuffer);
      releaseBuffer(gradValueBuffer);
    }
  }
}
