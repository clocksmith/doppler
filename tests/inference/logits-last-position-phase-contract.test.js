import assert from 'node:assert/strict';

const {
  resolveLmHeadMatmulConfig,
} = await import('../../src/inference/pipelines/text/logits/index.js');

assert.deepEqual(
  resolveLmHeadMatmulConfig(182, { lastPositionOnly: true }),
  {
    lastPositionOnly: true,
    matmulRows: 1,
    phaseOverride: 'decode',
  }
);

assert.deepEqual(
  resolveLmHeadMatmulConfig(1, { lastPositionOnly: true }),
  {
    lastPositionOnly: false,
    matmulRows: 1,
    phaseOverride: null,
  }
);

assert.deepEqual(
  resolveLmHeadMatmulConfig(182, null),
  {
    lastPositionOnly: false,
    matmulRows: 182,
    phaseOverride: null,
  }
);

console.log('logits-last-position-phase-contract.test: ok');
