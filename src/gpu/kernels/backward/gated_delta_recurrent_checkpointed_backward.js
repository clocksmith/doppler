import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { runGatedDeltaRecurrentBackward } from './gated_delta_recurrent_backward.js';
import { runGatedDeltaRecurrentCheckpointForward } from './gated_delta_recurrent_checkpoint_forward.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function validate(inputs, options) {
  const totalTokens = positiveInteger(options?.totalTokens, 'totalTokens');
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const keyDim = positiveInteger(options?.keyDim, 'keyDim');
  const valueDim = positiveInteger(options?.valueDim, 'valueDim');
  const checkpointInterval = Math.min(
    positiveInteger(options?.checkpointInterval, 'checkpointInterval'),
    totalTokens
  );
  const queryScale = Number(options?.queryScale);
  if (valueDim > 128 || !Number.isFinite(queryScale)) {
    throw new Error('checkpointed backward requires valueDim <= 128 and finite queryScale.');
  }
  for (const [label, tensor] of Object.entries(inputs)) {
    if (tensor?.dtype !== 'f32') {
      throw new Error(`checkpointed backward requires f32 ${label}.`);
    }
  }
  return { totalTokens, numHeads, keyDim, valueDim, checkpointInterval, queryScale };
}

export async function runGatedDeltaRecurrentCheckpointedBackward(inputs, options = {}) {
  const dims = validate(inputs, options);
  const queryBytes = dims.totalTokens * dims.numHeads * dims.keyDim * 4;
  const valueBytes = dims.totalTokens * dims.numHeads * dims.valueDim * 4;
  const scalarBytes = dims.totalTokens * dims.numHeads * 4;
  const stateElements = dims.numHeads * dims.keyDim * dims.valueDim;
  const stateBytes = stateElements * 4;
  const blockCount = Math.ceil(dims.totalTokens / dims.checkpointInterval);
  const gradQueryBuffer = acquireBuffer(queryBytes, undefined, 'checkpointed_gated_delta_grad_query');
  const gradKeyBuffer = acquireBuffer(queryBytes, undefined, 'checkpointed_gated_delta_grad_key');
  const gradValueBuffer = acquireBuffer(valueBytes, undefined, 'checkpointed_gated_delta_grad_value');
  const gradLogDecayBuffer = acquireBuffer(scalarBytes, undefined, 'checkpointed_gated_delta_grad_log_decay');
  const gradBetaBuffer = acquireBuffer(scalarBytes, undefined, 'checkpointed_gated_delta_grad_beta');
  const gradStateBuffer = acquireBuffer(stateBytes, undefined, 'checkpointed_gated_delta_grad_state');
  const recomputeStateBuffer = acquireBuffer(stateBytes, undefined, 'checkpointed_gated_delta_recompute_state');
  const recomputeOutputBuffer = acquireBuffer(valueBytes, undefined, 'checkpointed_gated_delta_recompute_output');
  let result = null;
  let completed = false;
  try {
    for (let block = blockCount - 1; block >= 0; block -= 1) {
      const tokenOffset = block * dims.checkpointInterval;
      const numTokens = Math.min(dims.checkpointInterval, dims.totalTokens - tokenOffset);
      const recomputed = await runGatedDeltaRecurrentCheckpointForward({
        query: inputs.query,
        key: inputs.key,
        value: inputs.value,
        logDecay: inputs.logDecay,
        beta: inputs.beta,
        initialState: inputs.checkpoints,
      }, {
        numTokens,
        totalTokens: dims.totalTokens,
        tokenOffset,
        numHeads: dims.numHeads,
        keyDim: dims.keyDim,
        valueDim: dims.valueDim,
        checkpointInterval: 1,
        initialStateOffsetElements: block * stateElements,
        queryScale: dims.queryScale,
        stateBuffer: recomputeStateBuffer,
        outputBuffer: recomputeOutputBuffer,
      });
      try {
        result = await runGatedDeltaRecurrentBackward({
          query: inputs.query,
          key: inputs.key,
          value: inputs.value,
          logDecay: inputs.logDecay,
          beta: inputs.beta,
          stateHistory: recomputed.checkpoints,
          gradOutput: inputs.gradOutput,
        }, {
          numTokens,
          totalTokens: dims.totalTokens,
          tokenOffset,
          numHeads: dims.numHeads,
          keyDim: dims.keyDim,
          valueDim: dims.valueDim,
          queryScale: dims.queryScale,
          initializeOutputBuffers: block === blockCount - 1,
          initializeGradState: block === blockCount - 1,
          gradQueryBuffer,
          gradKeyBuffer,
          gradValueBuffer,
          gradLogDecayBuffer,
          gradBetaBuffer,
          gradStateBuffer,
        });
      } finally {
        releaseBuffer(recomputed.checkpoints.buffer);
      }
    }
    completed = true;
    return { ...result, blockCount, checkpointInterval: dims.checkpointInterval };
  } finally {
    releaseBuffer(recomputeStateBuffer);
    releaseBuffer(recomputeOutputBuffer);
    if (!completed) {
      releaseBuffer(gradQueryBuffer);
      releaseBuffer(gradKeyBuffer);
      releaseBuffer(gradValueBuffer);
      releaseBuffer(gradLogDecayBuffer);
      releaseBuffer(gradBetaBuffer);
      releaseBuffer(gradStateBuffer);
    }
  }
}
