import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  causalConvSiluBackward,
  causalConvSiluForward,
  gatedRmsNormBackward,
  gatedRmsNormForward,
} from '../../../src/experimental/training/qwen-linear-attention-reference.js';
import {
  gatedDeltaRecurrentBackward,
  gatedDeltaRecurrentForward,
} from '../../../src/experimental/training/qwen-gated-delta-reference.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import {
  runCausalConv1dSiluBackward,
  runGatedDeltaRecurrentBackward,
  runGatedRmsNormBackward,
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
  try {
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
    return { actual, comparison: compare(actual, expected) };
  } finally {
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
  try {
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
      gradInput: compare(actualInput, expected.input),
      gradGate: compare(actualGate, expected.gate),
    };
  } finally {
    if (result?.gradInput?.buffer) releaseBuffer(result.gradInput.buffer);
    if (result?.gradGate?.buffer) releaseBuffer(result.gradGate.buffer);
    releaseBuffer(input.buffer);
    releaseBuffer(gate.buffer);
    releaseBuffer(weight.buffer);
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
  const expected = gatedDeltaRecurrentBackward(inputs, gradOutputValues, forward.cache, options);
  const tensors = {
    query: makeTensor(inputs.query, [options.numTokens, options.numHeads, options.keyDim], 'gated_delta_query'),
    key: makeTensor(inputs.key, [options.numTokens, options.numHeads, options.keyDim], 'gated_delta_key'),
    value: makeTensor(inputs.value, [options.numTokens, options.numHeads, options.valueDim], 'gated_delta_value'),
    logDecay: makeTensor(inputs.logDecay, [options.numTokens, options.numHeads], 'gated_delta_log_decay'),
    beta: makeTensor(inputs.beta, [options.numTokens, options.numHeads], 'gated_delta_beta'),
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
  try {
    result = await runGatedDeltaRecurrentBackward(tensors, options);
    const comparisons = {};
    for (const key of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
      comparisons[key] = compare(await readF32(result[key]), expected[key]);
    }
    return comparisons;
  } finally {
    if (result) {
      for (const tensor of Object.values(result)) {
        if (tensor?.buffer) releaseBuffer(tensor.buffer);
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
  const gatedDeltaRecurrent = await runGatedDeltaRecurrentCase();
  const comparisons = {
    causalConvGradInput: causalConv.comparison,
    gatedRmsNormGradInput: gatedRmsNorm.gradInput,
    gatedRmsNormGradGate: gatedRmsNorm.gradGate,
    gatedDeltaGradQuery: gatedDeltaRecurrent.query,
    gatedDeltaGradKey: gatedDeltaRecurrent.key,
    gatedDeltaGradValue: gatedDeltaRecurrent.value,
    gatedDeltaGradLogDecay: gatedDeltaRecurrent.logDecay,
    gatedDeltaGradBeta: gatedDeltaRecurrent.beta,
    gatedDeltaGradInitialState: gatedDeltaRecurrent.initialState,
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
    claimBoundary: 'Full-history recurrent, causal-convolution input, and gated-RMSNorm input/gate GPU gradients only; production checkpoint/recompute and Qwen layer integration remain absent.',
  };
}
