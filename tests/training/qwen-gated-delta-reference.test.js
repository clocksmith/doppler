import assert from 'node:assert/strict';

import {
  estimateGatedDeltaCheckpointElements,
  gatedDeltaRecurrentBackward,
  gatedDeltaRecurrentCheckpointedBackward,
  gatedDeltaRecurrentCheckpointedForward,
  gatedDeltaRecurrentForward,
} from '../../src/experimental/training/qwen-gated-delta-reference.js';

const options = {
  numTokens: 3,
  numHeads: 2,
  keyDim: 2,
  valueDim: 3,
  queryScale: 1 / Math.sqrt(2),
};

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.71) * scale
  );
}

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
const gradOutput = values(options.numTokens * options.numHeads * options.valueDim, 31, 0.6);
const forward = gatedDeltaRecurrentForward(inputs, options);
const analytic = gatedDeltaRecurrentBackward(inputs, gradOutput, forward.cache, options);

function objective(candidate) {
  const result = gatedDeltaRecurrentForward(candidate, options).output;
  let total = 0;
  for (let index = 0; index < result.length; index += 1) {
    total += result[index] * gradOutput[index];
  }
  return total;
}

const epsilon = 1e-3;
const tolerance = 2e-4;
for (const inputName of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
  assert.equal(analytic[inputName].length, inputs[inputName].length);
  for (let index = 0; index < inputs[inputName].length; index += 1) {
    const plus = { ...inputs, [inputName]: new Float32Array(inputs[inputName]) };
    const minus = { ...inputs, [inputName]: new Float32Array(inputs[inputName]) };
    plus[inputName][index] += epsilon;
    minus[inputName][index] -= epsilon;
    const numeric = (objective(plus) - objective(minus)) / (2 * epsilon);
    const error = Math.abs(analytic[inputName][index] - numeric);
    assert.ok(
      error <= tolerance,
      `${inputName}[${index}] analytic=${analytic[inputName][index]} numeric=${numeric} error=${error}`
    );
  }
}

assert.deepEqual(
  Array.from(forward.finalState),
  Array.from(forward.cache.states.slice(
    options.numTokens * options.numHeads * options.keyDim * options.valueDim
  ))
);

assert.throws(
  () => gatedDeltaRecurrentForward({ ...inputs, beta: new Float32Array(1) }, options),
  /beta must be Float32Array/
);

for (const checkpointInterval of [1, 2, 3]) {
  const checkpointOptions = { ...options, checkpointInterval };
  const checkpointed = gatedDeltaRecurrentCheckpointedForward(inputs, checkpointOptions);
  assert.deepEqual(Array.from(checkpointed.output), Array.from(forward.output));
  assert.deepEqual(Array.from(checkpointed.finalState), Array.from(forward.finalState));
  const checkpointedGradients = gatedDeltaRecurrentCheckpointedBackward(
    inputs,
    gradOutput,
    checkpointed.cache,
    checkpointOptions
  );
  for (const key of ['query', 'key', 'value', 'logDecay', 'beta', 'initialState']) {
    assert.deepEqual(Array.from(checkpointedGradients[key]), Array.from(analytic[key]));
  }
}

const productionEstimate = estimateGatedDeltaCheckpointElements({
  numTokens: 640,
  numHeads: 32,
  keyDim: 128,
  valueDim: 128,
  queryScale: 1 / Math.sqrt(128),
  checkpointInterval: 32,
});
assert.equal(productionEstimate.stateElements, 524288);
assert.equal(productionEstimate.blockCount, 20);
assert.equal(productionEstimate.storedCheckpointElements, 11010048);
assert.equal(productionEstimate.peakRecomputedBlockElements, 17301504);
assert.ok(productionEstimate.peakBackwardStateElements < productionEstimate.fullHistoryElements);

assert.throws(
  () => gatedDeltaRecurrentCheckpointedForward(inputs, { ...options, checkpointInterval: 0 }),
  /checkpointInterval must be a positive integer/
);

console.log('qwen-gated-delta-reference.test: ok');
