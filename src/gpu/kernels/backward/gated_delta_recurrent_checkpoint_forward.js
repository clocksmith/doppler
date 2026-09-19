import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
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

function validate(inputs, options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const totalTokens = positiveInteger(options?.totalTokens, 'totalTokens');
  const tokenOffset = Math.floor(Number(options?.tokenOffset));
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const keyDim = positiveInteger(options?.keyDim, 'keyDim');
  const valueDim = positiveInteger(options?.valueDim, 'valueDim');
  const checkpointInterval = positiveInteger(options?.checkpointInterval, 'checkpointInterval');
  const initialStateOffsetElements = Math.floor(Number(options?.initialStateOffsetElements));
  const queryScale = Number(options?.queryScale);
  if (!Number.isInteger(tokenOffset) || tokenOffset < 0 || tokenOffset + numTokens > totalTokens) {
    throw new Error('tokenOffset must address numTokens within totalTokens.');
  }
  if (!Number.isInteger(initialStateOffsetElements) || initialStateOffsetElements < 0) {
    throw new Error('initialStateOffsetElements must be a non-negative integer.');
  }
  if (valueDim > 128 || !Number.isFinite(queryScale)) {
    throw new Error('checkpoint forward requires valueDim <= 128 and finite queryScale.');
  }
  for (const [label, tensor] of Object.entries(inputs)) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`checkpoint forward requires f32 ${label}.`);
    }
  }
  return {
    numTokens,
    totalTokens,
    tokenOffset,
    numHeads,
    keyDim,
    valueDim,
    checkpointInterval: Math.min(checkpointInterval, numTokens),
    initialStateOffsetElements,
    queryScale,
  };
}

export async function runGatedDeltaRecurrentCheckpointForward(inputs, options = {}) {
  const dims = validate(inputs, options);
  const device = getDevice();
  if (!device) throw new Error('checkpoint forward requires an active GPU device.');
  const queryBytes = dims.totalTokens * dims.numHeads * dims.keyDim * 4;
  const valueBytes = dims.totalTokens * dims.numHeads * dims.valueDim * 4;
  const scalarBytes = dims.totalTokens * dims.numHeads * 4;
  const stateBytes = dims.numHeads * dims.keyDim * dims.valueDim * 4;
  const checkpointCount = Math.ceil(dims.numTokens / dims.checkpointInterval);
  const packedQueryKeyBuffer = acquireBuffer(queryBytes * 2, undefined, 'gated_delta_forward_query_key');
  const packedDecayBetaBuffer = acquireBuffer(scalarBytes * 2, undefined, 'gated_delta_forward_decay_beta');
  const stateBuffer = options.stateBuffer
    || acquireBuffer(stateBytes, undefined, 'gated_delta_forward_state');
  const checkpointBuffer = options.checkpointBuffer
    || acquireBuffer((checkpointCount + 1) * stateBytes, undefined, 'gated_delta_forward_checkpoints');
  const outputBuffer = options.outputBuffer
    || acquireBuffer(valueBytes, undefined, 'gated_delta_forward_output');
  let uniformBuffer = null;
  let completed = false;
  try {
    const packEncoder = device.createCommandEncoder({ label: 'gated_delta_forward_pack_inputs' });
    packEncoder.copyBufferToBuffer(inputs.query.buffer, 0, packedQueryKeyBuffer, 0, queryBytes);
    packEncoder.copyBufferToBuffer(inputs.key.buffer, 0, packedQueryKeyBuffer, queryBytes, queryBytes);
    packEncoder.copyBufferToBuffer(inputs.logDecay.buffer, 0, packedDecayBetaBuffer, 0, scalarBytes);
    packEncoder.copyBufferToBuffer(inputs.beta.buffer, 0, packedDecayBetaBuffer, scalarBytes, scalarBytes);
    device.queue.submit([packEncoder.finish()]);

    const pipeline = await createPipeline('gated_delta_recurrent_checkpoint_forward', 'default');
    uniformBuffer = createUniformBufferWithView(
      'gated_delta_recurrent_checkpoint_forward_uniforms',
      48,
      (view) => {
        view.setUint32(0, dims.numTokens, true);
        view.setUint32(4, dims.totalTokens, true);
        view.setUint32(8, dims.tokenOffset, true);
        view.setUint32(12, dims.numHeads, true);
        view.setUint32(16, dims.keyDim, true);
        view.setUint32(20, dims.valueDim, true);
        view.setUint32(24, dims.checkpointInterval, true);
        view.setUint32(28, checkpointCount, true);
        view.setFloat32(32, dims.queryScale, true);
        view.setUint32(36, dims.initialStateOffsetElements, true);
      },
      null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'gated_delta_recurrent_checkpoint_forward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: packedQueryKeyBuffer } },
        { binding: 2, resource: { buffer: inputs.value.buffer } },
        { binding: 3, resource: { buffer: packedDecayBetaBuffer } },
        { binding: 4, resource: { buffer: inputs.initialState.buffer } },
        { binding: 5, resource: { buffer: stateBuffer } },
        { binding: 6, resource: { buffer: checkpointBuffer } },
        { binding: 7, resource: { buffer: outputBuffer } },
      ],
    });
    dispatch(device, pipeline, bindGroup, dims.numHeads, 'gated_delta_recurrent_checkpoint_forward');
    await device.queue.onSubmittedWorkDone();
    completed = true;
    return {
      output: createTensor(
        outputBuffer,
        'f32',
        [dims.totalTokens, dims.numHeads, dims.valueDim],
        'gated_delta_checkpoint_output'
      ),
      checkpoints: createTensor(
        checkpointBuffer,
        'f32',
        [checkpointCount + 1, dims.numHeads, dims.keyDim, dims.valueDim],
        'gated_delta_checkpoints'
      ),
      finalState: createTensor(
        stateBuffer,
        'f32',
        [dims.numHeads, dims.keyDim, dims.valueDim],
        'gated_delta_final_state'
      ),
      checkpointCount,
    };
  } finally {
    await device.queue.onSubmittedWorkDone().catch(() => {});
    if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
    releaseBuffer(packedQueryKeyBuffer);
    releaseBuffer(packedDecayBetaBuffer);
    if (!completed) {
      if (!options.stateBuffer) releaseBuffer(stateBuffer);
      if (!options.checkpointBuffer) releaseBuffer(checkpointBuffer);
      if (!options.outputBuffer) releaseBuffer(outputBuffer);
    }
  }
}
