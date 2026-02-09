
import { getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { createPipeline, createUniformBufferWithView, createBindGroupWithValidation } from './utils.js';
import { dispatchKernel } from './dispatch.js';
import { trace } from '../../debug/index.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { getDevice } from '../device.js';

function selectSoftmaxVariant(innerSize) {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  const { smallThreshold } = getKernelThresholds().softmax;
  const isSmall = innerSize <= smallThreshold;
  return selectRuleValue('softmax', 'variant', { hasSubgroups, isSmall });
}

async function _softmax(target, input, axis, options = {}) {
  const { batchSize = 1, size, seqLen, temperature = 1.0, outputBuffer = null } = options;

  const bytesPerElement = input.dtype === 'f16' ? 2 : 4;
  const inferredSize = size || seqLen || (input.buffer.size / (batchSize * bytesPerElement));
  const variant = selectSoftmaxVariant(inferredSize);
  trace.kernels(`Softmax: size=${inferredSize}, variant=${variant}`);

  const outputSize = batchSize * inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'softmax_output');

  await unifiedKernelWrapper(
    'softmax', target, variant,
    [input, output],
    { inner_size: inferredSize, outer_size: batchSize, temperature },
    batchSize
  );

  return createTensor(output, input.dtype, [batchSize, inferredSize], 'softmax_output');
}

export async function runSoftmax(input, axis, options = {}) {
  return _softmax(null, input, axis, options);
}

export async function recordSoftmax(recorder, input, axis, options = {}) {
  return _softmax(recorder, input, axis, options);
}

export async function runSoftmaxTopK(logits, numTokens, numExperts, topK, options = {}) {
  const device = getDevice();
  const { normalize = true, inputDtype = 'f32', weightsDtype = 'f32' } = options;

  if (weightsDtype === 'f16' && inputDtype !== 'f16') {
    throw new Error('SoftmaxTopK f16 weights require f16 logits');
  }

  const variant = selectRuleValue('softmax', 'topkVariant', { inputDtype, weightsDtype });
  const pipeline = await createPipeline('topk', variant);

  const indicesSize = numTokens * topK * 4;
  const weightsBytesPerElement = weightsDtype === 'f16' ? 2 : 4;
  const weightsSize = numTokens * topK * weightsBytesPerElement;

  const indices = acquireBuffer(indicesSize, undefined, 'softmax_topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'softmax_topk_weights');

  const uniformBuffer = createUniformBufferWithView(
    'softmax_topk_uniforms', 16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null, device
  );

  const bindGroup = await createBindGroupWithValidation(device, {
    label: 'softmax_topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  }, `topk:${variant}`);

  dispatchKernel(null, pipeline, bindGroup, numTokens, 'softmax_topk');
  uniformBuffer.destroy();

  return { indices, weights };
}
