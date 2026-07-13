import assert from 'node:assert/strict';

import {
  qwenAttentionSplitQGateBackward,
  qwenAttentionSplitQGateForward,
  sigmoidGateBackward,
  sigmoidGateForward,
  partialRopeBackward,
  partialRopeForward,
  qwenFullAttentionModuleBackward,
  qwenFullAttentionModuleForward,
} from '../../src/experimental/training/qwen-full-attention-reference.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.47) * scale
  );
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

{
  const options = { numTokens: 2, numHeads: 3, headDim: 4 };
  const input = values(options.numTokens * options.numHeads * options.headDim * 2, 3, 0.4);
  const gradQuery = values(input.length / 2, 17, 0.3);
  const gradGate = values(input.length / 2, 29, 0.25);
  const forward = qwenAttentionSplitQGateForward(input, options);
  assert.deepEqual(
    qwenAttentionSplitQGateBackward(gradQuery, gradGate, options),
    Float32Array.from({ length: input.length }, (_, index) => {
      const column = index % (options.headDim * 2);
      const row = Math.floor(index / (options.headDim * 2));
      const source = (row * options.headDim) + (column % options.headDim);
      return column < options.headDim ? gradQuery[source] : gradGate[source];
    })
  );
  const epsilon = 1e-3;
  for (let index = 0; index < input.length; index += 1) {
    const plus = new Float32Array(input);
    const minus = new Float32Array(input);
    plus[index] += epsilon;
    minus[index] -= epsilon;
    const plusOutput = qwenAttentionSplitQGateForward(plus, options);
    const minusOutput = qwenAttentionSplitQGateForward(minus, options);
    const numeric = (
      dot(plusOutput.query, gradQuery)
      + dot(plusOutput.gate, gradGate)
      - dot(minusOutput.query, gradQuery)
      - dot(minusOutput.gate, gradGate)
    ) / (2 * epsilon);
    const analytic = qwenAttentionSplitQGateBackward(gradQuery, gradGate, options)[index];
    assert.ok(Math.abs(analytic - numeric) <= 2e-5);
  }
  assert.equal(forward.query.length, input.length / 2);
}

{
  const input = values(12, 7, 0.5);
  const gate = values(12, 19, 0.4);
  const gradOutput = values(12, 31, 0.3);
  const analytic = sigmoidGateBackward(input, gate, gradOutput);
  const epsilon = 1e-3;
  for (const [name, valuesForInput] of [['input', input], ['gate', gate]]) {
    for (let index = 0; index < valuesForInput.length; index += 1) {
      const plusInput = new Float32Array(input);
      const minusInput = new Float32Array(input);
      const plusGate = new Float32Array(gate);
      const minusGate = new Float32Array(gate);
      const plus = name === 'input' ? plusInput : plusGate;
      const minus = name === 'input' ? minusInput : minusGate;
      plus[index] += epsilon;
      minus[index] -= epsilon;
      const numeric = (
        dot(sigmoidGateForward(plusInput, plusGate), gradOutput)
        - dot(sigmoidGateForward(minusInput, minusGate), gradOutput)
      ) / (2 * epsilon);
      assert.ok(Math.abs(analytic[name][index] - numeric) <= 2e-5);
    }
  }
}

{
  const options = {
    numTokens: 3,
    numHeads: 2,
    headDim: 8,
    rotaryDim: 4,
    pairSpanDim: 4,
    interleaved: true,
    startPos: 0,
  };
  const input = values(options.numTokens * options.numHeads * options.headDim, 5, 0.4);
  const gradOutput = values(input.length, 37, 0.3);
  const cos = Float32Array.from({ length: options.numTokens * 2 }, (_, index) => Math.cos(index * 0.17));
  const sin = Float32Array.from({ length: options.numTokens * 2 }, (_, index) => Math.sin(index * 0.17));
  const analytic = partialRopeBackward(gradOutput, cos, sin, options);
  const epsilon = 1e-3;
  for (let index = 0; index < input.length; index += 1) {
    const plus = new Float32Array(input);
    const minus = new Float32Array(input);
    plus[index] += epsilon;
    minus[index] -= epsilon;
    const numeric = (
      dot(partialRopeForward(plus, cos, sin, options), gradOutput)
      - dot(partialRopeForward(minus, cos, sin, options), gradOutput)
    ) / (2 * epsilon);
    assert.ok(Math.abs(analytic[index] - numeric) <= 2e-5);
  }
}

{
  const options = {
    numTokens: 2,
    hiddenSize: 3,
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
  const fixed = {
    qWeight: values(querySize * 2 * options.hiddenSize, 3, 0.2),
    kWeight: values(kvSize * options.hiddenSize, 53, 0.18),
    vWeight: values(kvSize * options.hiddenSize, 67, 0.19),
    oWeight: values(options.hiddenSize * querySize, 79, 0.2),
    qNormWeight: values(options.headDim, 103, 0.08),
    kNormWeight: values(options.headDim, 109, 0.08),
    cos: Float32Array.from({ length: options.numTokens }, (_, index) => Math.cos(index * 0.17)),
    sin: Float32Array.from({ length: options.numTokens }, (_, index) => Math.sin(index * 0.17)),
  };
  const hidden = values(options.numTokens * options.hiddenSize, 113, 0.25);
  const gradOutput = values(options.numTokens * options.hiddenSize, 127, 0.3);
  const forward = qwenFullAttentionModuleForward({ ...fixed, hidden }, options);
  const analytic = qwenFullAttentionModuleBackward(
    { ...fixed, hidden },
    gradOutput,
    forward.cache,
    options
  ).hidden;
  const epsilon = 1e-4;
  for (let index = 0; index < hidden.length; index += 1) {
    const plus = new Float32Array(hidden);
    const minus = new Float32Array(hidden);
    plus[index] += epsilon;
    minus[index] -= epsilon;
    const numeric = (
      dot(qwenFullAttentionModuleForward({ ...fixed, hidden: plus }, options).output, gradOutput)
      - dot(qwenFullAttentionModuleForward({ ...fixed, hidden: minus }, options).output, gradOutput)
    ) / (2 * epsilon);
    assert.ok(
      Math.abs(analytic[index] - numeric) <= 3e-4,
      `full attention hidden[${index}] analytic=${analytic[index]} numeric=${numeric}`
    );
  }
}

{
  const options = {
    numTokens: 2,
    hiddenSize: 3,
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
  const lora = {
    q: {
      A: values(options.hiddenSize * rank, 137, 0.12),
      B: values(rank * querySize * 2, 149, 0.1),
      rank,
      alpha: 4,
    },
    k: {
      A: values(options.hiddenSize * rank, 181, 0.11),
      B: values(rank * kvSize, 193, 0.09),
      rank,
      alpha: 4,
    },
    v: {
      A: values(options.hiddenSize * rank, 211, 0.1),
      B: values(rank * kvSize, 223, 0.08),
      rank,
      alpha: 4,
    },
    o: {
      A: values(querySize * rank, 239, 0.09),
      B: values(rank * options.hiddenSize, 277, 0.1),
      rank,
      alpha: 4,
    },
  };
  const inputs = {
    hidden: values(options.numTokens * options.hiddenSize, 293, 0.25),
    qWeight: values(querySize * 2 * options.hiddenSize, 307, 0.2),
    kWeight: values(kvSize * options.hiddenSize, 359, 0.18),
    vWeight: values(kvSize * options.hiddenSize, 373, 0.19),
    oWeight: values(options.hiddenSize * querySize, 389, 0.2),
    qNormWeight: values(options.headDim, 419, 0.08),
    kNormWeight: values(options.headDim, 431, 0.08),
    cos: Float32Array.from(
      { length: options.numTokens },
      (_, index) => Math.cos(index * 0.17)
    ),
    sin: Float32Array.from(
      { length: options.numTokens },
      (_, index) => Math.sin(index * 0.17)
    ),
    lora,
  };
  const gradOutput = values(options.numTokens * options.hiddenSize, 443, 0.3);
  const forward = qwenFullAttentionModuleForward(inputs, options);
  const analytic = qwenFullAttentionModuleBackward(
    inputs,
    gradOutput,
    forward.cache,
    options
  );
  const epsilon = 1e-4;
  for (const projection of ['q', 'k', 'v', 'o']) {
    for (const matrix of ['A', 'B']) {
      const parameter = lora[projection][matrix];
      for (let index = 0; index < parameter.length; index += 1) {
        const original = parameter[index];
        parameter[index] = original + epsilon;
        const plus = dot(qwenFullAttentionModuleForward(inputs, options).output, gradOutput);
        parameter[index] = original - epsilon;
        const minus = dot(qwenFullAttentionModuleForward(inputs, options).output, gradOutput);
        parameter[index] = original;
        const numeric = (plus - minus) / (2 * epsilon);
        const actual = analytic.lora[projection][matrix][index];
        assert.ok(
          Math.abs(actual - numeric) <= 6e-4,
          `${projection}.${matrix}[${index}] analytic=${actual} numeric=${numeric}`
        );
      }
    }
  }
}

console.log('qwen-full-attention-reference.test: ok');
