import assert from 'node:assert/strict';

import {
  computeSiluGatedBackwardValues,
  OpType,
} from '../../src/experimental/training/autograd.js';
import { loadBackwardRegistry } from '../../src/config/backward-registry-loader.js';

function silu(value) {
  const clamped = Math.max(-15, Math.min(15, value));
  return value / (1 + Math.exp(-clamped));
}

function loss(gate, up, gradOutput) {
  let total = 0;
  for (let index = 0; index < gate.length; index += 1) {
    total += silu(gate[index]) * up[index] * gradOutput[index];
  }
  return total;
}

const gate = new Float32Array([-1.25, -0.1, 0.75, 2.0]);
const up = new Float32Array([0.3, -0.7, 1.1, -0.2]);
const gradOutput = new Float32Array([0.5, -0.25, 0.75, 1.25]);
const analytic = computeSiluGatedBackwardValues(gate, up, gradOutput);
const epsilon = 1e-3;

for (let index = 0; index < gate.length; index += 1) {
  const plusGate = new Float32Array(gate);
  const minusGate = new Float32Array(gate);
  plusGate[index] += epsilon;
  minusGate[index] -= epsilon;
  const numericGate = (
    loss(plusGate, up, gradOutput) - loss(minusGate, up, gradOutput)
  ) / (2 * epsilon);
  assert.ok(Math.abs(analytic.gradGate[index] - numericGate) < 2e-4);

  const plusUp = new Float32Array(up);
  const minusUp = new Float32Array(up);
  plusUp[index] += epsilon;
  minusUp[index] -= epsilon;
  const numericUp = (
    loss(gate, plusUp, gradOutput) - loss(gate, minusUp, gradOutput)
  ) / (2 * epsilon);
  assert.ok(Math.abs(analytic.gradUp[index] - numericUp) < 2e-4);
}

const limited = computeSiluGatedBackwardValues(
  new Float32Array([4]),
  new Float32Array([4]),
  new Float32Array([1]),
  1
);
assert.equal(limited.gradGate[0], 0);
assert.equal(limited.gradUp[0], 0);

const registry = loadBackwardRegistry();
assert.equal(OpType.SILU_GATED, 'silu_gated');
assert.equal(registry.ops.silu_gated.backward, 'silu_gated_backward');
assert.deepEqual(registry.ops.silu_gated.grads, ['gate', 'up']);

console.log('silu-gated-autograd-contract.test: ok');
