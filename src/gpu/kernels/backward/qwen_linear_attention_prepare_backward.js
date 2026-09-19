import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { dispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

const WORKGROUP_SIZE = 128;

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function requireElements(tensor, expected, label) {
  if (tensor?.dtype !== 'f32') {
    throw new Error(`Qwen linear-attention prepare backward requires f32 ${label}.`);
  }
  const actual = tensor.shape?.reduce((product, value) => product * value, 1);
  if (actual !== expected) {
    throw new Error(
      `Qwen linear-attention prepare backward requires ${label} to have ${expected} elements.`
    );
  }
}

function resolveDimensions(inputs, options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const numKeyHeads = positiveInteger(options?.numKeyHeads, 'numKeyHeads');
  const numValueHeads = positiveInteger(options?.numValueHeads, 'numValueHeads');
  const keyDim = positiveInteger(options?.keyDim, 'keyDim');
  const valueDim = positiveInteger(options?.valueDim, 'valueDim');
  const eps = Number(options?.eps);
  if (numValueHeads % numKeyHeads !== 0) {
    throw new Error('numValueHeads must be divisible by numKeyHeads.');
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('eps must be finite and positive.');
  }
  const querySize = numKeyHeads * keyDim;
  const keySize = querySize;
  const valueSize = numValueHeads * valueDim;
  const convSize = querySize + keySize + valueSize;
  const queryElements = numTokens * numValueHeads * keyDim;
  const valueElements = numTokens * valueSize;
  const scalarElements = numTokens * numValueHeads;
  for (const [label, expected] of [
    ['mixed', numTokens * convSize],
    ['a', scalarElements],
    ['b', scalarElements],
    ['aLog', numValueHeads],
    ['dtBias', numValueHeads],
    ['gradQuery', queryElements],
    ['gradKey', queryElements],
    ['gradValue', valueElements],
    ['gradLogDecay', scalarElements],
    ['gradBeta', scalarElements],
  ]) {
    requireElements(inputs[label], expected, label);
  }
  return {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    eps,
    querySize,
    keySize,
    valueSize,
    convSize,
    repeatFactor: numValueHeads / numKeyHeads,
    queryElements,
    valueElements,
    scalarElements,
  };
}

export async function runQwenLinearAttentionPrepareBackward(inputs, options = {}) {
  const dims = resolveDimensions(inputs, options);
  const device = getDevice();
  if (!device) {
    throw new Error('Qwen linear-attention prepare backward requires an active GPU device.');
  }
  const elementBytes = Float32Array.BYTES_PER_ELEMENT;
  const mixedBytes = dims.numTokens * dims.convSize * elementBytes;
  const queryBytes = dims.queryElements * elementBytes;
  const valueBytes = dims.valueElements * elementBytes;
  const scalarBytes = dims.scalarElements * elementBytes;
  const headBytes = dims.numValueHeads * elementBytes;
  const packedAB = acquireBuffer(scalarBytes * 2, undefined, 'qwen_linear_prepare_backward_a_b');
  const packedParameters = acquireBuffer(
    headBytes * 2,
    undefined,
    'qwen_linear_prepare_backward_parameters'
  );
  const packedGradQueryKey = acquireBuffer(
    queryBytes * 2,
    undefined,
    'qwen_linear_prepare_backward_grad_query_key'
  );
  const packedGradDecayBeta = acquireBuffer(
    scalarBytes * 2,
    undefined,
    'qwen_linear_prepare_backward_grad_decay_beta'
  );
  const gradMixedBuffer = acquireBuffer(mixedBytes, undefined, 'qwen_linear_prepare_grad_mixed');
  const gradABuffer = acquireBuffer(scalarBytes, undefined, 'qwen_linear_prepare_grad_a');
  const gradBBuffer = acquireBuffer(scalarBytes, undefined, 'qwen_linear_prepare_grad_b');
  let uniformBuffer = null;
  let completed = false;
  try {
    const packEncoder = device.createCommandEncoder({ label: 'qwen_linear_prepare_backward_pack' });
    packEncoder.copyBufferToBuffer(inputs.a.buffer, 0, packedAB, 0, scalarBytes);
    packEncoder.copyBufferToBuffer(inputs.b.buffer, 0, packedAB, scalarBytes, scalarBytes);
    packEncoder.copyBufferToBuffer(inputs.aLog.buffer, 0, packedParameters, 0, headBytes);
    packEncoder.copyBufferToBuffer(inputs.dtBias.buffer, 0, packedParameters, headBytes, headBytes);
    packEncoder.copyBufferToBuffer(inputs.gradQuery.buffer, 0, packedGradQueryKey, 0, queryBytes);
    packEncoder.copyBufferToBuffer(
      inputs.gradKey.buffer,
      0,
      packedGradQueryKey,
      queryBytes,
      queryBytes
    );
    packEncoder.copyBufferToBuffer(
      inputs.gradLogDecay.buffer,
      0,
      packedGradDecayBeta,
      0,
      scalarBytes
    );
    packEncoder.copyBufferToBuffer(
      inputs.gradBeta.buffer,
      0,
      packedGradDecayBeta,
      scalarBytes,
      scalarBytes
    );
    device.queue.submit([packEncoder.finish()]);

    const pipeline = await createPipeline('qwen_linear_attention_prepare_backward', 'default');
    uniformBuffer = createUniformBufferWithView(
      'qwen_linear_attention_prepare_backward_uniforms',
      48,
      (view) => {
        view.setUint32(0, dims.numTokens, true);
        view.setUint32(4, dims.numKeyHeads, true);
        view.setUint32(8, dims.numValueHeads, true);
        view.setUint32(12, dims.keyDim, true);
        view.setUint32(16, dims.valueDim, true);
        view.setUint32(20, dims.convSize, true);
        view.setUint32(24, dims.querySize, true);
        view.setUint32(28, dims.keySize, true);
        view.setUint32(32, dims.valueSize, true);
        view.setUint32(36, dims.repeatFactor, true);
        view.setFloat32(40, dims.eps, true);
      },
      null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'qwen_linear_attention_prepare_backward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputs.mixed.buffer } },
        { binding: 2, resource: { buffer: packedAB } },
        { binding: 3, resource: { buffer: packedParameters } },
        { binding: 4, resource: { buffer: packedGradQueryKey } },
        { binding: 5, resource: { buffer: inputs.gradValue.buffer } },
        { binding: 6, resource: { buffer: packedGradDecayBeta } },
        { binding: 7, resource: { buffer: gradMixedBuffer } },
        { binding: 8, resource: { buffer: gradABuffer } },
        { binding: 9, resource: { buffer: gradBBuffer } },
      ],
    });
    const sourceRows = dims.numTokens * dims.numKeyHeads;
    const linearElements = dims.valueElements + (dims.scalarElements * 2);
    const workgroups = (sourceRows * 2) + Math.ceil(linearElements / WORKGROUP_SIZE);
    dispatch(device, pipeline, bindGroup, workgroups, 'qwen_linear_attention_prepare_backward');
    await device.queue.onSubmittedWorkDone();
    completed = true;
    return {
      mixed: createTensor(
        gradMixedBuffer,
        'f32',
        [dims.numTokens, dims.convSize],
        'qwen_linear_prepare_grad_mixed'
      ),
      a: createTensor(
        gradABuffer,
        'f32',
        [dims.numTokens, dims.numValueHeads],
        'qwen_linear_prepare_grad_a'
      ),
      b: createTensor(
        gradBBuffer,
        'f32',
        [dims.numTokens, dims.numValueHeads],
        'qwen_linear_prepare_grad_b'
      ),
    };
  } finally {
    await device.queue.onSubmittedWorkDone().catch(() => {});
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    releaseBuffer(packedAB);
    releaseBuffer(packedParameters);
    releaseBuffer(packedGradQueryKey);
    releaseBuffer(packedGradDecayBeta);
    if (!completed) {
      releaseBuffer(gradMixedBuffer);
      releaseBuffer(gradABuffer);
      releaseBuffer(gradBBuffer);
    }
  }
}
