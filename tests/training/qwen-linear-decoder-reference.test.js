import assert from 'node:assert/strict';

import {
  qwenLinearDecoderLayerBackward,
  qwenLinearDecoderLayerForward,
} from '../../src/experimental/training/qwen-linear-decoder-reference.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.39) * scale
  );
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

const options = {
  numTokens: 3,
  hiddenSize: 4,
  intermediateSize: 6,
  numKeyHeads: 1,
  numValueHeads: 2,
  keyDim: 2,
  valueDim: 2,
  kernelSize: 2,
  checkpointInterval: 2,
  queryScale: 1 / Math.sqrt(2),
  l2Eps: 1e-6,
  rmsEps: 1e-6,
};
const convSize = (options.numKeyHeads * options.keyDim * 2)
  + (options.numValueHeads * options.valueDim);
const valueSize = options.numValueHeads * options.valueDim;
const rank = 2;
const adapter = (inputSize, outputSize, offset) => ({
  A: values(inputSize * rank, offset, 0.08),
  B: values(rank * outputSize, offset + 13, 0.06),
  rank,
  alpha: 4,
});
const inputs = {
  hidden: values(options.numTokens * options.hiddenSize, 3, 0.2),
  inputNormWeight: values(options.hiddenSize, 23, 0.05),
  postAttentionNormWeight: values(options.hiddenSize, 31, 0.05),
  attention: {
    qkvWeight: values(convSize * options.hiddenSize, 43, 0.12),
    zWeight: values(valueSize * options.hiddenSize, 79, 0.11),
    aWeight: values(options.numValueHeads * options.hiddenSize, 97, 0.1),
    bWeight: values(options.numValueHeads * options.hiddenSize, 109, 0.1),
    outWeight: values(options.hiddenSize * valueSize, 127, 0.12),
    convWeight: values(convSize * options.kernelSize, 149, 0.1),
    aLog: values(options.numValueHeads, 173, 0.1),
    dtBias: values(options.numValueHeads, 181, 0.08),
    normWeight: values(options.valueDim, 193, 0.05),
    initialState: values(
      options.numValueHeads * options.keyDim * options.valueDim,
      211,
      0.07
    ),
  },
  mlp: {
    gateWeight: values(options.intermediateSize * options.hiddenSize, 229, 0.13),
    upWeight: values(options.intermediateSize * options.hiddenSize, 257, 0.12),
    downWeight: values(options.hiddenSize * options.intermediateSize, 283, 0.11),
    lora: {
      gate: adapter(options.hiddenSize, options.intermediateSize, 311),
      up: adapter(options.hiddenSize, options.intermediateSize, 347),
      down: adapter(options.intermediateSize, options.hiddenSize, 379),
    },
  },
};
const gradOutput = values(options.numTokens * options.hiddenSize, 419, 0.23);
const forward = qwenLinearDecoderLayerForward(inputs, options);
const analytic = qwenLinearDecoderLayerBackward(inputs, gradOutput, forward.cache, options);
const epsilon = 1e-4;

function numericGradient(parameter, index) {
  const original = parameter[index];
  parameter[index] = original + epsilon;
  const plus = dot(qwenLinearDecoderLayerForward(inputs, options).output, gradOutput);
  parameter[index] = original - epsilon;
  const minus = dot(qwenLinearDecoderLayerForward(inputs, options).output, gradOutput);
  parameter[index] = original;
  return (plus - minus) / (2 * epsilon);
}

for (let index = 0; index < inputs.hidden.length; index += 1) {
  const numeric = numericGradient(inputs.hidden, index);
  assert.ok(
    Math.abs(analytic.hidden[index] - numeric) <= 1e-3,
    `hidden[${index}] analytic=${analytic.hidden[index]} numeric=${numeric}`
  );
}

for (let index = 0; index < inputs.attention.initialState.length; index += 1) {
  const numeric = numericGradient(inputs.attention.initialState, index);
  assert.ok(
    Math.abs(analytic.initialState[index] - numeric) <= 1e-3,
    `initialState[${index}] analytic=${analytic.initialState[index]} numeric=${numeric}`
  );
}

for (const [name, lora] of Object.entries(inputs.mlp.lora)) {
  for (const matrix of ['A', 'B']) {
    for (let index = 0; index < lora[matrix].length; index += 1) {
      const numeric = numericGradient(lora[matrix], index);
      const actual = analytic.lora[name][matrix][index];
      assert.ok(
        Math.abs(actual - numeric) <= 1e-3,
        `${name}.${matrix}[${index}] analytic=${actual} numeric=${numeric}`
      );
    }
  }
}

console.log('qwen-linear-decoder-reference.test: ok');
