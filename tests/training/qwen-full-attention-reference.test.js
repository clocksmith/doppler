import assert from 'node:assert/strict';

import {
  qwenAttentionSplitQGateBackward,
  qwenAttentionSplitQGateForward,
  sigmoidGateBackward,
  sigmoidGateForward,
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

console.log('qwen-full-attention-reference.test: ok');
