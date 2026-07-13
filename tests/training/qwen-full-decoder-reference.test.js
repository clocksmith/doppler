import assert from 'node:assert/strict';

import {
  qwenFullDecoderLayerBackward,
  qwenFullDecoderLayerForward,
} from '../../src/experimental/training/qwen-full-decoder-reference.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.37) * scale
  );
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

const options = {
  numTokens: 2,
  hiddenSize: 3,
  intermediateSize: 5,
  numHeads: 2,
  numKVHeads: 1,
  headDim: 4,
  rotaryDim: 2,
  pairSpanDim: 2,
  interleaved: true,
  startPos: 0,
  rmsEps: 1e-6,
};
const querySize = options.numHeads * options.headDim;
const kvSize = options.numKVHeads * options.headDim;
const rank = 2;
const adapter = (inputSize, outputSize, offset) => ({
  A: values(inputSize * rank, offset, 0.09),
  B: values(rank * outputSize, offset + 17, 0.07),
  rank,
  alpha: 4,
});
const inputs = {
  hidden: values(options.numTokens * options.hiddenSize, 3, 0.2),
  inputNormWeight: values(options.hiddenSize, 29, 0.05),
  postAttentionNormWeight: values(options.hiddenSize, 37, 0.05),
  attention: {
    qWeight: values(querySize * 2 * options.hiddenSize, 43, 0.12),
    kWeight: values(kvSize * options.hiddenSize, 101, 0.11),
    vWeight: values(kvSize * options.hiddenSize, 127, 0.1),
    oWeight: values(options.hiddenSize * querySize, 149, 0.12),
    qNormWeight: values(options.headDim, 181, 0.05),
    kNormWeight: values(options.headDim, 193, 0.05),
    cos: Float32Array.from(
      { length: options.numTokens * (options.rotaryDim / 2) },
      (_, index) => Math.cos(index * 0.13)
    ),
    sin: Float32Array.from(
      { length: options.numTokens * (options.rotaryDim / 2) },
      (_, index) => Math.sin(index * 0.13)
    ),
    lora: {
      q: adapter(options.hiddenSize, querySize * 2, 211),
      k: adapter(options.hiddenSize, kvSize, 251),
      v: adapter(options.hiddenSize, kvSize, 283),
      o: adapter(querySize, options.hiddenSize, 313),
    },
  },
  mlp: {
    gateWeight: values(options.intermediateSize * options.hiddenSize, 347, 0.13),
    upWeight: values(options.intermediateSize * options.hiddenSize, 373, 0.12),
    downWeight: values(options.hiddenSize * options.intermediateSize, 401, 0.11),
    lora: {
      gate: adapter(options.hiddenSize, options.intermediateSize, 431),
      up: adapter(options.hiddenSize, options.intermediateSize, 461),
      down: adapter(options.intermediateSize, options.hiddenSize, 491),
    },
  },
};
const gradOutput = values(options.numTokens * options.hiddenSize, 523, 0.25);
const forward = qwenFullDecoderLayerForward(inputs, options);
const analytic = qwenFullDecoderLayerBackward(inputs, gradOutput, forward.cache, options);
const epsilon = 1e-4;

for (let index = 0; index < inputs.hidden.length; index += 1) {
  const original = inputs.hidden[index];
  inputs.hidden[index] = original + epsilon;
  const plus = dot(qwenFullDecoderLayerForward(inputs, options).output, gradOutput);
  inputs.hidden[index] = original - epsilon;
  const minus = dot(qwenFullDecoderLayerForward(inputs, options).output, gradOutput);
  inputs.hidden[index] = original;
  const numeric = (plus - minus) / (2 * epsilon);
  assert.ok(
    Math.abs(analytic.hidden[index] - numeric) <= 8e-4,
    `hidden[${index}] analytic=${analytic.hidden[index]} numeric=${numeric}`
  );
}

for (const [name, lora] of [
  ...Object.entries(inputs.attention.lora),
  ...Object.entries(inputs.mlp.lora),
]) {
  for (const matrix of ['A', 'B']) {
    for (let index = 0; index < lora[matrix].length; index += 1) {
      const original = lora[matrix][index];
      lora[matrix][index] = original + epsilon;
      const plus = dot(qwenFullDecoderLayerForward(inputs, options).output, gradOutput);
      lora[matrix][index] = original - epsilon;
      const minus = dot(qwenFullDecoderLayerForward(inputs, options).output, gradOutput);
      lora[matrix][index] = original;
      const numeric = (plus - minus) / (2 * epsilon);
      const actual = analytic.lora[name][matrix][index];
      assert.ok(
        Math.abs(actual - numeric) <= 8e-4,
        `${name}.${matrix}[${index}] analytic=${actual} numeric=${numeric}`
      );
    }
  }
}

console.log('qwen-full-decoder-reference.test: ok');
