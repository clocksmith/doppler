import assert from 'node:assert/strict';

const { applyLinearNormWeightOffset } = await import('../../src/inference/pipelines/text/linear-attention.js');

{
  const values = new Float32Array([0.25, -0.5, 1.0]);
  const result = applyLinearNormWeightOffset(values, false);
  assert.equal(result, values);
  assert.deepEqual(Array.from(result), [0.25, -0.5, 1.0]);
}

{
  const values = new Float32Array([0.25, -0.5, 1.0]);
  const result = applyLinearNormWeightOffset(values, true);
  assert.notEqual(result, values);
  assert.deepEqual(Array.from(values), [0.25, -0.5, 1.0]);
  assert.deepEqual(Array.from(result), [1.25, 0.5, 2.0]);
}

console.log('qwen-linear-norm-offset.test: ok');
