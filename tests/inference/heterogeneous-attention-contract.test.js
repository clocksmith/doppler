import assert from 'node:assert/strict';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/manifest.schema.js';
import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';
import { isRoPEDisabledForLayer } from '../../src/inference/pipelines/text/attention/heterogeneous-contract.js';
import { projectSeparateAttentionGate } from '../../src/inference/pipelines/text/attention/gate-projection.js';

const valid = structuredClone(DEFAULT_MANIFEST_INFERENCE);
valid.attention.queryScale = 3.87;
valid.rope.disabledLayers = [3, 7, 11];
assert.doesNotThrow(() => validateRequiredInferenceFields(valid, 'heterogeneous-attention-fixture'));
assert.equal(isRoPEDisabledForLayer({ ropeDisabledLayers: valid.rope.disabledLayers }, 3), true);
assert.equal(isRoPEDisabledForLayer({ ropeDisabledLayers: valid.rope.disabledLayers }, 4), false);

for (const queryScale of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, '3.87']) {
  const invalid = structuredClone(valid);
  invalid.attention.queryScale = queryScale;
  assert.throws(
    () => validateRequiredInferenceFields(invalid, 'invalid-query-scale'),
    /queryScale must be a positive finite number/
  );
}

const duplicateNoPeLayer = structuredClone(valid);
duplicateNoPeLayer.rope.disabledLayers = [3, 3];
assert.throws(
  () => validateRequiredInferenceFields(duplicateNoPeLayer, 'duplicate-nope-layer'),
  /unique non-negative integer/
);

const matmulCalls = [];
const projectedGate = { buffer: { label: 'gate-output' }, dtype: 'f32' };
const projectionInput = { buffer: { label: 'attention-input' }, dtype: 'f32' };
const gateWeight = { buffer: { label: 'gate-weight' }, dtype: 'f32' };
assert.equal(await projectSeparateAttentionGate({
  runMatmul: async (...args) => {
    matmulCalls.push(args);
    return projectedGate;
  },
  projectionInput,
  gateWeight,
  numTokens: 2,
  outputSize: 8,
  hiddenSize: 16,
  layerIdx: 3,
}), projectedGate);
assert.deepEqual(matmulCalls[0].slice(0, 5), [projectionInput, gateWeight, 2, 8, 16]);
assert.equal(matmulCalls[0][5].role, 'q_gate_proj');

console.log('heterogeneous-attention-contract.test: ok');
