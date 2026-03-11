import assert from 'node:assert/strict';

const { inferLinearNormMode } = await import('../../src/inference/pipelines/text/linear-attention.js');

const projectionLayout = {
  headVDim: 128,
  valueDim: 2048,
};

{
  const sharedMode = inferLinearNormMode(
    { size: projectionLayout.headVDim * Float32Array.BYTES_PER_ELEMENT, dtype: 'f32' },
    projectionLayout
  );
  assert.equal(sharedMode, 'shared');
}

{
  const perHeadMode = inferLinearNormMode(
    { size: projectionLayout.valueDim * Float32Array.BYTES_PER_ELEMENT, dtype: 'f32' },
    projectionLayout
  );
  assert.equal(perHeadMode, 'per_head');
}

{
  const f16SharedMode = inferLinearNormMode(
    { size: projectionLayout.headVDim * Uint16Array.BYTES_PER_ELEMENT, dtype: 'f16' },
    projectionLayout
  );
  assert.equal(f16SharedMode, 'shared');
}

console.log('linear-attention-contract.test: ok');
