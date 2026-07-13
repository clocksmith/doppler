import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  causalConvSiluBackward,
  causalConvSiluForward,
  gatedRmsNormBackward,
  gatedRmsNormForward,
} from '../../../src/experimental/training/qwen-linear-attention-reference.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import {
  runCausalConv1dSiluBackward,
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
  const comparisons = {
    causalConvGradInput: causalConv.comparison,
    gatedRmsNormGradInput: gatedRmsNorm.gradInput,
    gatedRmsNormGradGate: gatedRmsNorm.gradGate,
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
    claimBoundary: 'Causal-convolution input and gated-RMSNorm input/gate GPU gradients only; recurrent gated-delta GPU backward and Qwen layer integration remain absent.',
  };
}
