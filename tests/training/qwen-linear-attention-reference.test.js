import assert from 'node:assert/strict';

import {
  causalConvSiluBackward,
  causalConvSiluForward,
  gatedDeltaParametersBackward,
  gatedDeltaParametersForward,
  gatedRmsNormBackward,
  gatedRmsNormForward,
  l2NormalizeBackward,
  l2NormalizeForward,
} from '../../src/experimental/training/qwen-linear-attention-reference.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.61) * scale
  );
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

function checkFiniteDifference({ inputs, analytic, objective, epsilon = 1e-3, tolerance = 2e-4 }) {
  for (const [name, valuesForInput] of Object.entries(inputs)) {
    for (let index = 0; index < valuesForInput.length; index += 1) {
      const plus = Object.fromEntries(
        Object.entries(inputs).map(([key, value]) => [key, new Float32Array(value)])
      );
      const minus = Object.fromEntries(
        Object.entries(inputs).map(([key, value]) => [key, new Float32Array(value)])
      );
      plus[name][index] += epsilon;
      minus[name][index] -= epsilon;
      const numeric = (objective(plus) - objective(minus)) / (2 * epsilon);
      const error = Math.abs(analytic[name][index] - numeric);
      assert.ok(
        error <= tolerance,
        `${name}[${index}] analytic=${analytic[name][index]} numeric=${numeric} error=${error}`
      );
    }
  }
}

{
  const options = { numTokens: 4, channels: 3, kernelSize: 3 };
  const inputs = {
    input: values(options.numTokens * options.channels, 1, 0.4),
    weight: values(options.channels * options.kernelSize, 9, 0.3),
  };
  const gradOutput = values(inputs.input.length, 17, 0.5);
  const forward = causalConvSiluForward(inputs.input, inputs.weight, options);
  const analytic = causalConvSiluBackward(
    inputs.input,
    inputs.weight,
    gradOutput,
    forward.cache,
    options
  );
  checkFiniteDifference({
    inputs,
    analytic,
    objective: (candidate) => dot(
      causalConvSiluForward(candidate.input, candidate.weight, options).output,
      gradOutput
    ),
  });
}

{
  const options = { rows: 3, width: 4, eps: 1e-6 };
  const inputs = {
    input: values(options.rows * options.width, 3, 0.5),
    gate: values(options.rows * options.width, 13, 0.4),
    weight: Float32Array.from(values(options.width, 23, 0.2), (value) => 1 + value),
  };
  const gradOutput = values(inputs.input.length, 29, 0.45);
  const forward = gatedRmsNormForward(inputs.input, inputs.gate, inputs.weight, options);
  const analytic = gatedRmsNormBackward(
    inputs.input,
    inputs.gate,
    inputs.weight,
    gradOutput,
    forward.cache,
    options
  );
  checkFiniteDifference({
    inputs,
    analytic,
    objective: (candidate) => dot(
      gatedRmsNormForward(candidate.input, candidate.gate, candidate.weight, options).output,
      gradOutput
    ),
  });
}

{
  const options = { rows: 4, width: 3, eps: 1e-6 };
  const input = values(options.rows * options.width, 5, 0.55);
  const gradOutput = values(input.length, 19, 0.35);
  const forward = l2NormalizeForward(input, options);
  const analytic = { input: l2NormalizeBackward(input, gradOutput, forward.cache, options) };
  checkFiniteDifference({
    inputs: { input },
    analytic,
    objective: (candidate) => dot(l2NormalizeForward(candidate.input, options).output, gradOutput),
  });
}

{
  const inputs = {
    a: values(6, 2, 0.4),
    b: values(6, 7, 0.3),
    aLog: values(6, 13, 0.2),
    dtBias: values(6, 19, 0.25),
  };
  const gradLogDecay = values(6, 23, 0.35);
  const gradBeta = values(6, 29, 0.45);
  const analytic = gatedDeltaParametersBackward(
    inputs.a,
    inputs.b,
    inputs.aLog,
    inputs.dtBias,
    gradLogDecay,
    gradBeta
  );
  checkFiniteDifference({
    inputs,
    analytic,
    objective: (candidate) => {
      const forward = gatedDeltaParametersForward(
        candidate.a,
        candidate.b,
        candidate.aLog,
        candidate.dtBias
      );
      return dot(forward.logDecay, gradLogDecay) + dot(forward.beta, gradBeta);
    },
  });
}

assert.throws(
  () => causalConvSiluForward(new Float32Array(1), new Float32Array(1), {
    numTokens: 0,
    channels: 1,
    kernelSize: 1,
  }),
  /numTokens must be a positive integer/
);

console.log('qwen-linear-attention-reference.test: ok');
