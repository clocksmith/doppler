import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  causalConvSiluBackward,
  causalConvSiluForward,
  gatedRmsNormBackward,
  gatedRmsNormForward,
  qwenLinearAttentionPrepareBackward,
  qwenLinearAttentionPrepareForward,
  qwenLinearAttentionCoreBackward,
  qwenLinearAttentionCoreForward,
} from '../../../src/experimental/training/qwen-linear-attention-reference.js';
import {
  releaseQwenLinearAttentionTrainingCoreCache,
  runQwenLinearAttentionTrainingCoreBackward,
  runQwenLinearAttentionTrainingCoreForward,
} from '../../../src/experimental/training/qwen-linear-attention-training-core.js';
import {
  gatedDeltaRecurrentBackward,
  gatedDeltaRecurrentCheckpointedBackward,
  gatedDeltaRecurrentCheckpointedForward,
  gatedDeltaRecurrentForward,
} from '../../../src/experimental/training/qwen-gated-delta-reference.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import {
  runCausalConv1dSilu,
  runGatedRmsNorm,
  runQwenLinearAttentionPrepare,
} from '../../../src/gpu/kernels/index.js';
import {
  runCausalConv1dSiluBackward,
  runGatedDeltaRecurrentBackward,
  runGatedDeltaRecurrentCheckpointForward,
  runGatedDeltaRecurrentCheckpointedBackward,
  runGatedRmsNormBackward,
  runQwenLinearAttentionPrepareBackward,
} from '../../../src/gpu/kernels/backward/index.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function makeTensor(values, shape, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return createTensor(buffer, 'f32', shape, label);
}

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.61) * scale
  );
}

function compare(actual, expected) {
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * Float32Array.BYTES_PER_ELEMENT));
}

async function runCausalConvCase(gradOffset, label) {
  const options = { numTokens: 4, channels: 3, kernelSize: 3 };
  const inputValues = values(options.numTokens * options.channels, 1, 0.4);
  const weightValues = values(options.channels * options.kernelSize, 9, 0.3);
  const gradValues = values(inputValues.length, gradOffset, 0.5);
  const input = makeTensor(inputValues, [options.numTokens, options.channels], `${label}_input`);
  const weight = makeTensor(weightValues, [options.channels, options.kernelSize], `${label}_weight`);
  const gradOutput = makeTensor(gradValues, [options.numTokens, options.channels], `${label}_grad_output`);
  let result = null;
  let forwardResult = null;
  try {
    forwardResult = await runCausalConv1dSilu(input, weight, options);
    result = await runCausalConv1dSiluBackward(input, weight, gradOutput, options);
    const actual = await readF32(result);
    const forward = causalConvSiluForward(inputValues, weightValues, options);
    const expected = causalConvSiluBackward(
      inputValues,
      weightValues,
      gradValues,
      forward.cache,
      options
    ).input;
    return {
      actual,
      backwardComparison: compare(actual, expected),
      forwardComparison: compare(await readF32(forwardResult), forward.output),
    };
  } finally {
    if (forwardResult?.buffer) releaseBuffer(forwardResult.buffer);
    if (result?.buffer) releaseBuffer(result.buffer);
    releaseBuffer(input.buffer);
    releaseBuffer(weight.buffer);
    releaseBuffer(gradOutput.buffer);
  }
}

async function runGatedRmsNormCase() {
  const options = { rows: 3, width: 4, eps: 1e-6 };
  const inputValues = values(options.rows * options.width, 3, 0.5);
  const gateValues = values(options.rows * options.width, 13, 0.4);
  const weightValues = Float32Array.from(values(options.width, 23, 0.2), (value) => 1 + value);
  const gradValues = values(inputValues.length, 29, 0.45);
  const input = makeTensor(inputValues, [options.rows, options.width], 'gated_rms_input');
  const gate = makeTensor(gateValues, [options.rows, options.width], 'gated_rms_gate');
  const weight = makeTensor(weightValues, [options.width], 'gated_rms_weight');
  const gradOutput = makeTensor(gradValues, [options.rows, options.width], 'gated_rms_grad_output');
  let result = null;
  let forwardResult = null;
  try {
    forwardResult = await runGatedRmsNorm(input, gate, weight, options);
    result = await runGatedRmsNormBackward(input, gate, weight, gradOutput, options);
    const actualInput = await readF32(result.gradInput);
    const actualGate = await readF32(result.gradGate);
    const forward = gatedRmsNormForward(inputValues, gateValues, weightValues, options);
    const expected = gatedRmsNormBackward(
      inputValues,
      gateValues,
      weightValues,
      gradValues,
      forward.cache,
      options
    );
    return {
      forward: compare(await readF32(forwardResult), forward.output),
      gradInput: compare(actualInput, expected.input),
      gradGate: compare(actualGate, expected.gate),
    };
  } finally {
    if (forwardResult?.buffer) releaseBuffer(forwardResult.buffer);
    if (result?.gradInput?.buffer) releaseBuffer(result.gradInput.buffer);
    if (result?.gradGate?.buffer) releaseBuffer(result.gradGate.buffer);
    releaseBuffer(input.buffer);
    releaseBuffer(gate.buffer);
    releaseBuffer(weight.buffer);
    releaseBuffer(gradOutput.buffer);
  }
}

async function runPrepareCase() {
  const options = {
    numTokens: 3,
    numKeyHeads: 2,
    numValueHeads: 4,
    keyDim: 2,
    valueDim: 3,
    eps: 1e-6,
  };
  const convSize = (options.numKeyHeads * options.keyDim * 2)
    + (options.numValueHeads * options.valueDim);
  const inputValues = {
    mixed: values(options.numTokens * convSize, 7, 0.45),
    a: values(options.numTokens * options.numValueHeads, 71, 0.3),
    b: values(options.numTokens * options.numValueHeads, 83, 0.25),
    aLog: values(options.numValueHeads, 97, 0.2),
    dtBias: values(options.numValueHeads, 101, 0.15),
  };
  const gradientValues = {
    query: values(options.numTokens * options.numValueHeads * options.keyDim, 107, 0.35),
    key: values(options.numTokens * options.numValueHeads * options.keyDim, 113, 0.3),
    value: values(options.numTokens * options.numValueHeads * options.valueDim, 127, 0.4),
    logDecay: values(options.numTokens * options.numValueHeads, 131, 0.25),
    beta: values(options.numTokens * options.numValueHeads, 139, 0.2),
  };
  const inputTensors = {
    mixed: makeTensor(inputValues.mixed, [options.numTokens, convSize], 'prepare_mixed'),
    a: makeTensor(inputValues.a, [options.numTokens, options.numValueHeads], 'prepare_a'),
    b: makeTensor(inputValues.b, [options.numTokens, options.numValueHeads], 'prepare_b'),
    aLog: makeTensor(inputValues.aLog, [options.numValueHeads], 'prepare_a_log'),
    dtBias: makeTensor(inputValues.dtBias, [options.numValueHeads], 'prepare_dt_bias'),
  };
  const gradientTensors = {
    gradQuery: makeTensor(
      gradientValues.query,
      [options.numTokens, options.numValueHeads, options.keyDim],
      'prepare_grad_query'
    ),
    gradKey: makeTensor(
      gradientValues.key,
      [options.numTokens, options.numValueHeads, options.keyDim],
      'prepare_grad_key'
    ),
    gradValue: makeTensor(
      gradientValues.value,
      [options.numTokens, options.numValueHeads, options.valueDim],
      'prepare_grad_value'
    ),
    gradLogDecay: makeTensor(
      gradientValues.logDecay,
      [options.numTokens, options.numValueHeads],
      'prepare_grad_log_decay'
    ),
    gradBeta: makeTensor(
      gradientValues.beta,
      [options.numTokens, options.numValueHeads],
      'prepare_grad_beta'
    ),
  };
  const expectedForward = qwenLinearAttentionPrepareForward(inputValues, options);
  const expectedBackward = qwenLinearAttentionPrepareBackward(
    inputValues,
    gradientValues,
    expectedForward.cache,
    options
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenLinearAttentionPrepare(
      inputTensors.mixed,
      inputTensors.a,
      inputTensors.b,
      inputTensors.aLog,
      inputTensors.dtBias,
      options
    );
    backward = await runQwenLinearAttentionPrepareBackward({
      ...inputTensors,
      ...gradientTensors,
    }, options);
    return {
      forwardQuery: compare(await readF32(forward.query), expectedForward.query),
      forwardKey: compare(await readF32(forward.key), expectedForward.key),
      forwardValue: compare(await readF32(forward.value), expectedForward.value),
      forwardLogDecay: compare(await readF32(forward.logDecay), expectedForward.logDecay),
      forwardBeta: compare(await readF32(forward.beta), expectedForward.beta),
      backwardMixed: compare(await readF32(backward.mixed), expectedBackward.mixed),
      backwardA: compare(await readF32(backward.a), expectedBackward.a),
      backwardB: compare(await readF32(backward.b), expectedBackward.b),
    };
  } finally {
    if (forward) {
      for (const tensor of Object.values(forward)) releaseBuffer(tensor.buffer);
    }
    if (backward) {
      for (const tensor of Object.values(backward)) releaseBuffer(tensor.buffer);
    }
    for (const tensor of Object.values(inputTensors)) releaseBuffer(tensor.buffer);
    for (const tensor of Object.values(gradientTensors)) releaseBuffer(tensor.buffer);
  }
}

async function runIntegratedCoreCase() {
  const options = {
    numTokens: 3,
    numKeyHeads: 1,
    numValueHeads: 2,
    keyDim: 2,
    valueDim: 3,
    kernelSize: 3,
    checkpointInterval: 2,
    queryScale: 1 / Math.sqrt(2),
    l2Eps: 1e-6,
    rmsEps: 1e-6,
  };
  const convSize = (options.numKeyHeads * options.keyDim * 2)
    + (options.numValueHeads * options.valueDim);
  const inputValues = {
    qkv: values(options.numTokens * convSize, 5, 0.3),
    z: values(options.numTokens * options.numValueHeads * options.valueDim, 41, 0.25),
    a: values(options.numTokens * options.numValueHeads, 67, 0.2),
    b: values(options.numTokens * options.numValueHeads, 73, 0.2),
    convWeight: values(convSize * options.kernelSize, 79, 0.2),
    aLog: values(options.numValueHeads, 113, 0.15),
    dtBias: values(options.numValueHeads, 127, 0.1),
    normWeight: Float32Array.from(
      values(options.valueDim, 131, 0.1),
      (value) => 1 + value
    ),
    initialState: values(
      options.numValueHeads * options.keyDim * options.valueDim,
      137,
      0.05
    ),
  };
  const gradOutputValues = values(
    options.numTokens * options.numValueHeads * options.valueDim,
    149,
    0.3
  );
  const tensors = {
    qkv: makeTensor(inputValues.qkv, [options.numTokens, convSize], 'core_qkv'),
    z: makeTensor(
      inputValues.z,
      [options.numTokens, options.numValueHeads, options.valueDim],
      'core_z'
    ),
    a: makeTensor(inputValues.a, [options.numTokens, options.numValueHeads], 'core_a'),
    b: makeTensor(inputValues.b, [options.numTokens, options.numValueHeads], 'core_b'),
    convWeight: makeTensor(
      inputValues.convWeight,
      [convSize, options.kernelSize],
      'core_conv_weight'
    ),
    aLog: makeTensor(inputValues.aLog, [options.numValueHeads], 'core_a_log'),
    dtBias: makeTensor(inputValues.dtBias, [options.numValueHeads], 'core_dt_bias'),
    normWeight: makeTensor(inputValues.normWeight, [options.valueDim], 'core_norm_weight'),
    initialState: makeTensor(
      inputValues.initialState,
      [options.numValueHeads, options.keyDim, options.valueDim],
      'core_initial_state'
    ),
  };
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.numTokens * options.numValueHeads, options.valueDim],
    'core_grad_output'
  );
  const referenceOptions = { ...options, eps: options.l2Eps };
  const expectedForward = qwenLinearAttentionCoreForward(inputValues, referenceOptions);
  const expectedBackward = qwenLinearAttentionCoreBackward(
    inputValues,
    gradOutputValues,
    expectedForward.cache,
    referenceOptions
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenLinearAttentionTrainingCoreForward(tensors, options);
    backward = await runQwenLinearAttentionTrainingCoreBackward(
      tensors,
      gradOutput,
      forward.cache,
      options
    );
    return {
      forwardOutput: compare(await readF32(forward.output), expectedForward.output),
      forwardFinalState: compare(await readF32(forward.finalState), expectedForward.finalState),
      backwardQkv: compare(await readF32(backward.qkv), expectedBackward.qkv),
      backwardZ: compare(await readF32(backward.z), expectedBackward.z),
      backwardA: compare(await readF32(backward.a), expectedBackward.a),
      backwardB: compare(await readF32(backward.b), expectedBackward.b),
      backwardInitialState: compare(
        await readF32(backward.initialState),
        expectedBackward.initialState
      ),
    };
  } finally {
    if (backward) {
      for (const tensor of Object.values(backward)) releaseBuffer(tensor.buffer);
    }
    if (forward) {
      releaseBuffer(forward.output.buffer);
      releaseBuffer(forward.finalState.buffer);
      releaseQwenLinearAttentionTrainingCoreCache(forward.cache);
    }
    for (const tensor of Object.values(tensors)) releaseBuffer(tensor.buffer);
    releaseBuffer(gradOutput.buffer);
  }
}

async function runGatedDeltaRecurrentCase() {
  const options = {
    numTokens: 3,
    numHeads: 2,
    keyDim: 2,
    valueDim: 3,
    queryScale: 1 / Math.sqrt(2),
  };
  const inputs = {
    query: values(options.numTokens * options.numHeads * options.keyDim, 1, 0.4),
    key: values(options.numTokens * options.numHeads * options.keyDim, 7, 0.35),
    value: values(options.numTokens * options.numHeads * options.valueDim, 13, 0.5),
    logDecay: values(options.numTokens * options.numHeads, 19, 0.12),
    beta: Float32Array.from(
      values(options.numTokens * options.numHeads, 23, 0.15),
      (value) => 0.55 + value
    ),
    initialState: values(options.numHeads * options.keyDim * options.valueDim, 29, 0.08),
  };
  const gradOutputValues = values(
    options.numTokens * options.numHeads * options.valueDim,
    31,
    0.6
  );
  const forward = gatedDeltaRecurrentForward(inputs, options);
  const checkpointInterval = 2;
  const checkpointedForward = gatedDeltaRecurrentCheckpointedForward(inputs, {
    ...options,
    checkpointInterval,
  });
  const checkpointedBackward = gatedDeltaRecurrentCheckpointedBackward(
    inputs,
    gradOutputValues,
    checkpointedForward.cache,
    { ...options, checkpointInterval }
  );
  const expected = gatedDeltaRecurrentBackward(inputs, gradOutputValues, forward.cache, options);
  const tensors = {
    query: makeTensor(inputs.query, [options.numTokens, options.numHeads, options.keyDim], 'gated_delta_query'),
    key: makeTensor(inputs.key, [options.numTokens, options.numHeads, options.keyDim], 'gated_delta_key'),
    value: makeTensor(inputs.value, [options.numTokens, options.numHeads, options.valueDim], 'gated_delta_value'),
    logDecay: makeTensor(inputs.logDecay, [options.numTokens, options.numHeads], 'gated_delta_log_decay'),
    beta: makeTensor(inputs.beta, [options.numTokens, options.numHeads], 'gated_delta_beta'),
    initialState: makeTensor(
      inputs.initialState,
      [options.numHeads, options.keyDim, options.valueDim],
      'gated_delta_initial_state'
    ),
    stateHistory: makeTensor(
      forward.cache.states,
      [options.numTokens + 1, options.numHeads, options.keyDim, options.valueDim],
      'gated_delta_state_history'
    ),
    gradOutput: makeTensor(
      gradOutputValues,
      [options.numTokens, options.numHeads, options.valueDim],
      'gated_delta_grad_output'
    ),
  };
  let result = null;
  let checkpointResult = null;
  let checkpointBackwardResult = null;
  try {
    checkpointResult = await runGatedDeltaRecurrentCheckpointForward({
      query: tensors.query,
      key: tensors.key,
      value: tensors.value,
      logDecay: tensors.logDecay,
      beta: tensors.beta,
      initialState: tensors.initialState,
    }, {
      ...options,
      totalTokens: options.numTokens,
      tokenOffset: 0,
      checkpointInterval,
      initialStateOffsetElements: 0,
    });
    checkpointBackwardResult = await runGatedDeltaRecurrentCheckpointedBackward({
      query: tensors.query,
      key: tensors.key,
      value: tensors.value,
      logDecay: tensors.logDecay,
      beta: tensors.beta,
      checkpoints: checkpointResult.checkpoints,
      gradOutput: tensors.gradOutput,
    }, {
      totalTokens: options.numTokens,
      numHeads: options.numHeads,
      keyDim: options.keyDim,
      valueDim: options.valueDim,
      checkpointInterval,
      queryScale: options.queryScale,
    });
    result = await runGatedDeltaRecurrentBackward({
      query: tensors.query,
      key: tensors.key,
      value: tensors.value,
      logDecay: tensors.logDecay,
      beta: tensors.beta,
      stateHistory: tensors.stateHistory,
      gradOutput: tensors.gradOutput,
    }, {
      ...options,
      totalTokens: options.numTokens,
      tokenOffset: 0,
      initializeOutputBuffers: true,
      initializeGradState: true,
    });
    const comparisons = {};
    for (const key of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
      comparisons[key] = compare(await readF32(result[key]), expected[key]);
    }
    comparisons.checkpointForwardOutput = compare(
      await readF32(checkpointResult.output),
      checkpointedForward.output
    );
    comparisons.checkpointForwardStates = compare(
      await readF32(checkpointResult.checkpoints),
      checkpointedForward.cache.checkpoints
    );
    comparisons.checkpointForwardFinalState = compare(
      await readF32(checkpointResult.finalState),
      checkpointedForward.finalState
    );
    for (const key of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
      comparisons[`checkpointBackward${key[0].toUpperCase()}${key.slice(1)}`] = compare(
        await readF32(checkpointBackwardResult[key]),
        checkpointedBackward[key]
      );
    }
    return comparisons;
  } finally {
    if (result) {
      for (const tensor of Object.values(result)) {
        if (tensor?.buffer) releaseBuffer(tensor.buffer);
      }
    }
    if (checkpointResult) {
      releaseBuffer(checkpointResult.output.buffer);
      releaseBuffer(checkpointResult.checkpoints.buffer);
      releaseBuffer(checkpointResult.finalState.buffer);
    }
    if (checkpointBackwardResult) {
      for (const key of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
        releaseBuffer(checkpointBackwardResult[key].buffer);
      }
    }
    for (const tensor of Object.values(tensors)) {
      releaseBuffer(tensor.buffer);
    }
  }
}

export async function runQwenLinearAttentionBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const tolerance = 2e-5;
  const causalConv = await runCausalConvCase(17, 'causal_conv_baseline');
  const perturbedCausalConv = await runCausalConvCase(31, 'causal_conv_perturbed');
  const causalConvPerturbation = compare(perturbedCausalConv.actual, causalConv.actual);
  const gatedRmsNorm = await runGatedRmsNormCase();
  const preparation = await runPrepareCase();
  const integratedCore = await runIntegratedCoreCase();
  const gatedDeltaRecurrent = await runGatedDeltaRecurrentCase();
  const comparisons = {
    causalConvForward: causalConv.forwardComparison,
    causalConvGradInput: causalConv.backwardComparison,
    gatedRmsNormForward: gatedRmsNorm.forward,
    gatedRmsNormGradInput: gatedRmsNorm.gradInput,
    gatedRmsNormGradGate: gatedRmsNorm.gradGate,
    prepareForwardQuery: preparation.forwardQuery,
    prepareForwardKey: preparation.forwardKey,
    prepareForwardValue: preparation.forwardValue,
    prepareForwardLogDecay: preparation.forwardLogDecay,
    prepareForwardBeta: preparation.forwardBeta,
    prepareBackwardMixed: preparation.backwardMixed,
    prepareBackwardA: preparation.backwardA,
    prepareBackwardB: preparation.backwardB,
    integratedCoreForwardOutput: integratedCore.forwardOutput,
    integratedCoreForwardFinalState: integratedCore.forwardFinalState,
    integratedCoreBackwardQkv: integratedCore.backwardQkv,
    integratedCoreBackwardZ: integratedCore.backwardZ,
    integratedCoreBackwardA: integratedCore.backwardA,
    integratedCoreBackwardB: integratedCore.backwardB,
    integratedCoreBackwardInitialState: integratedCore.backwardInitialState,
    gatedDeltaGradQuery: gatedDeltaRecurrent.query,
    gatedDeltaGradKey: gatedDeltaRecurrent.key,
    gatedDeltaGradValue: gatedDeltaRecurrent.value,
    gatedDeltaGradLogDecay: gatedDeltaRecurrent.logDecay,
    gatedDeltaGradBeta: gatedDeltaRecurrent.beta,
    gatedDeltaGradInitialState: gatedDeltaRecurrent.initialState,
    gatedDeltaCheckpointForwardOutput: gatedDeltaRecurrent.checkpointForwardOutput,
    gatedDeltaCheckpointForwardStates: gatedDeltaRecurrent.checkpointForwardStates,
    gatedDeltaCheckpointForwardFinalState: gatedDeltaRecurrent.checkpointForwardFinalState,
    gatedDeltaCheckpointBackwardQuery: gatedDeltaRecurrent.checkpointBackwardQuery,
    gatedDeltaCheckpointBackwardKey: gatedDeltaRecurrent.checkpointBackwardKey,
    gatedDeltaCheckpointBackwardValue: gatedDeltaRecurrent.checkpointBackwardValue,
    gatedDeltaCheckpointBackwardLogDecay: gatedDeltaRecurrent.checkpointBackwardLogDecay,
    gatedDeltaCheckpointBackwardBeta: gatedDeltaRecurrent.checkpointBackwardBeta,
    gatedDeltaCheckpointBackwardInitialState: gatedDeltaRecurrent.checkpointBackwardInitialState,
  };
  const passed = Object.values(comparisons).every(
    (entry) => entry.allFinite && entry.maxAbsError <= tolerance
  ) && causalConvPerturbation.maxAbsError > 1e-4;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'qwen_linear_attention_component_backward_oracle',
    schemaVersion: 1,
    passed,
    tolerance: { maxAbsError: tolerance },
    comparisons,
    negativeControl: {
      perturbation: 'replace_causal_conv_upstream_gradient_fixture',
      gradInputDifference: causalConvPerturbation,
      passed: causalConvPerturbation.maxAbsError > 1e-4,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Integrated checkpointed Qwen linear-attention core from projected QKV/Z/A/B through gated RMSNorm only; frozen input/output projection matmuls, residuals, and complete decoder-layer execution remain absent.',
  };
}
