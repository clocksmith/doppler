import {
  runCausalConv1dSilu,
  runGatedRmsNorm,
  runQwenLinearAttentionPrepare,
} from '../../gpu/kernels/index.js';
import {
  runCausalConv1dSiluBackward,
  runGatedDeltaRecurrentCheckpointForward,
  runGatedDeltaRecurrentCheckpointedBackward,
  runGatedRmsNormBackward,
  runQwenLinearAttentionPrepareBackward,
} from '../../gpu/kernels/backward/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function resolveDimensions(options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const numKeyHeads = positiveInteger(options?.numKeyHeads, 'numKeyHeads');
  const numValueHeads = positiveInteger(options?.numValueHeads, 'numValueHeads');
  const keyDim = positiveInteger(options?.keyDim, 'keyDim');
  const valueDim = positiveInteger(options?.valueDim, 'valueDim');
  const kernelSize = positiveInteger(options?.kernelSize, 'kernelSize');
  const checkpointInterval = positiveInteger(options?.checkpointInterval, 'checkpointInterval');
  const queryScale = Number(options?.queryScale);
  const l2Eps = Number(options?.l2Eps);
  const rmsEps = Number(options?.rmsEps);
  if (numValueHeads % numKeyHeads !== 0) {
    throw new Error('numValueHeads must be divisible by numKeyHeads.');
  }
  if (!Number.isFinite(queryScale)) {
    throw new Error('queryScale must be finite.');
  }
  if (!Number.isFinite(l2Eps) || l2Eps <= 0 || !Number.isFinite(rmsEps) || rmsEps <= 0) {
    throw new Error('l2Eps and rmsEps must be finite and positive.');
  }
  const querySize = numKeyHeads * keyDim;
  const valueSize = numValueHeads * valueDim;
  return {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    kernelSize,
    checkpointInterval: Math.min(checkpointInterval, numTokens),
    queryScale,
    l2Eps,
    rmsEps,
    convSize: (querySize * 2) + valueSize,
  };
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseTensorMap(tensors) {
  if (!tensors) return;
  for (const tensor of Object.values(tensors)) releaseTensor(tensor);
}

export async function runQwenLinearAttentionTrainingCoreForward(inputs, options = {}) {
  const dims = resolveDimensions(options);
  let convolution = null;
  let preparation = null;
  let recurrence = null;
  let output = null;
  let completed = false;
  try {
    convolution = await runCausalConv1dSilu(inputs.qkv, inputs.convWeight, {
      numTokens: dims.numTokens,
      channels: dims.convSize,
      kernelSize: dims.kernelSize,
    });
    preparation = await runQwenLinearAttentionPrepare(
      convolution,
      inputs.a,
      inputs.b,
      inputs.aLog,
      inputs.dtBias,
      {
        numTokens: dims.numTokens,
        numKeyHeads: dims.numKeyHeads,
        numValueHeads: dims.numValueHeads,
        keyDim: dims.keyDim,
        valueDim: dims.valueDim,
        eps: dims.l2Eps,
      }
    );
    recurrence = await runGatedDeltaRecurrentCheckpointForward({
      ...preparation,
      initialState: inputs.initialState,
    }, {
      numTokens: dims.numTokens,
      totalTokens: dims.numTokens,
      tokenOffset: 0,
      numHeads: dims.numValueHeads,
      keyDim: dims.keyDim,
      valueDim: dims.valueDim,
      checkpointInterval: dims.checkpointInterval,
      initialStateOffsetElements: 0,
      queryScale: dims.queryScale,
    });
    output = await runGatedRmsNorm(recurrence.output, inputs.z, inputs.normWeight, {
      rows: dims.numTokens * dims.numValueHeads,
      width: dims.valueDim,
      eps: dims.rmsEps,
    });
    completed = true;
    return {
      output,
      finalState: recurrence.finalState,
      cache: {
        dims,
        convolution,
        preparation,
        recurrenceOutput: recurrence.output,
        checkpoints: recurrence.checkpoints,
      },
    };
  } finally {
    if (!completed) {
      releaseTensor(output);
      if (recurrence) {
        releaseTensor(recurrence.output);
        releaseTensor(recurrence.checkpoints);
        releaseTensor(recurrence.finalState);
      }
      releaseTensorMap(preparation);
      releaseTensor(convolution);
    }
  }
}

export async function runQwenLinearAttentionTrainingCoreBackward(
  inputs,
  gradOutput,
  cache,
  options = {}
) {
  const dims = resolveDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`linear-attention training cache mismatch for ${key}.`);
    }
  }
  let normalizationGradients = null;
  let recurrenceGradients = null;
  let preparationGradients = null;
  let qkvGradient = null;
  let completed = false;
  try {
    normalizationGradients = await runGatedRmsNormBackward(
      cache.recurrenceOutput,
      inputs.z,
      inputs.normWeight,
      gradOutput,
      {
        rows: dims.numTokens * dims.numValueHeads,
        width: dims.valueDim,
        eps: dims.rmsEps,
      }
    );
    recurrenceGradients = await runGatedDeltaRecurrentCheckpointedBackward({
      ...cache.preparation,
      checkpoints: cache.checkpoints,
      gradOutput: normalizationGradients.gradInput,
    }, {
      totalTokens: dims.numTokens,
      numHeads: dims.numValueHeads,
      keyDim: dims.keyDim,
      valueDim: dims.valueDim,
      checkpointInterval: dims.checkpointInterval,
      queryScale: dims.queryScale,
    });
    preparationGradients = await runQwenLinearAttentionPrepareBackward({
      mixed: cache.convolution,
      a: inputs.a,
      b: inputs.b,
      aLog: inputs.aLog,
      dtBias: inputs.dtBias,
      gradQuery: recurrenceGradients.query,
      gradKey: recurrenceGradients.key,
      gradValue: recurrenceGradients.value,
      gradLogDecay: recurrenceGradients.logDecay,
      gradBeta: recurrenceGradients.beta,
    }, {
      numTokens: dims.numTokens,
      numKeyHeads: dims.numKeyHeads,
      numValueHeads: dims.numValueHeads,
      keyDim: dims.keyDim,
      valueDim: dims.valueDim,
      eps: dims.l2Eps,
    });
    qkvGradient = await runCausalConv1dSiluBackward(
      inputs.qkv,
      inputs.convWeight,
      preparationGradients.mixed,
      {
        numTokens: dims.numTokens,
        channels: dims.convSize,
        kernelSize: dims.kernelSize,
      }
    );
    completed = true;
    return {
      qkv: qkvGradient,
      z: normalizationGradients.gradGate,
      a: preparationGradients.a,
      b: preparationGradients.b,
      initialState: recurrenceGradients.initialState,
    };
  } finally {
    releaseTensor(normalizationGradients?.gradInput);
    if (recurrenceGradients) {
      for (const key of ['query', 'key', 'value', 'logDecay', 'beta']) {
        releaseTensor(recurrenceGradients[key]);
      }
    }
    releaseTensor(preparationGradients?.mixed);
    if (!completed) {
      releaseTensor(qkvGradient);
      releaseTensor(normalizationGradients?.gradGate);
      releaseTensor(preparationGradients?.a);
      releaseTensor(preparationGradients?.b);
      releaseTensor(recurrenceGradients?.initialState);
    }
  }
}

export function releaseQwenLinearAttentionTrainingCoreCache(cache) {
  releaseTensor(cache?.convolution);
  releaseTensorMap(cache?.preparation);
  releaseTensor(cache?.recurrenceOutput);
  releaseTensor(cache?.checkpoints);
}
