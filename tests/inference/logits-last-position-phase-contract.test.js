import assert from 'node:assert/strict';

const {
  resolveLmHeadMatmulConfig,
} = await import('../../src/inference/pipelines/text/logits/index.js');

assert.deepEqual(
  resolveLmHeadMatmulConfig(182, { lastPositionOnly: true }),
  {
    lastPositionOnly: true,
    trailingRows: null,
    matmulRows: 1,
    phaseOverride: 'prefill',
  }
);

assert.deepEqual(
  resolveLmHeadMatmulConfig(1, { lastPositionOnly: true }),
  {
    lastPositionOnly: false,
    trailingRows: null,
    matmulRows: 1,
    phaseOverride: null,
  }
);

assert.deepEqual(
  resolveLmHeadMatmulConfig(182, null),
  {
    lastPositionOnly: false,
    trailingRows: null,
    matmulRows: 182,
    phaseOverride: null,
  }
);

// Teacher-forced target scoring projects only the trailing rows it reads back.
assert.deepEqual(
  resolveLmHeadMatmulConfig(182, { trailingRows: 5 }),
  { lastPositionOnly: false, trailingRows: 5, matmulRows: 5, phaseOverride: 'prefill' }
);
assert.deepEqual(
  resolveLmHeadMatmulConfig(3, { trailingRows: 3 }),
  { lastPositionOnly: false, trailingRows: null, matmulRows: 3, phaseOverride: 'prefill' }
);
for (const bad of [0, 4, 1.5, -1]) {
  assert.throws(() => resolveLmHeadMatmulConfig(3, { trailingRows: bad }), /trailingRows must be/);
}
for (const conflict of [{ lastPositionOnly: true }, { selectedTokenIds: [1] }, { returnGpuBuffer: true }]) {
  assert.throws(() => resolveLmHeadMatmulConfig(3, { trailingRows: 2, ...conflict }), /excludes/);
}

console.log('logits-last-position-phase-contract.test: ok');
