import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

const defaultRuntime = createDopplerConfig().runtime;
assert.deepEqual(defaultRuntime.inference.largeWeights, {
  enabled: true,
  safetyRatio: 0.9,
  preferF16: true,
  lmHeadChunkRows: null,
});

const overriddenRuntime = createDopplerConfig({
  runtime: {
    inference: {
      largeWeights: {
        preferF16: false,
        lmHeadChunkRows: 512,
      },
    },
  },
}).runtime;

assert.deepEqual(overriddenRuntime.inference.largeWeights, {
  enabled: true,
  safetyRatio: 0.9,
  preferF16: false,
  lmHeadChunkRows: 512,
});

console.log('runtime-large-weights-defaults-contract.test: ok');
