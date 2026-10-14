import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

const defaultLargeWeights = createDopplerConfig().runtime.inference.largeWeights;

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
  ...defaultLargeWeights,
  preferF16: false,
  lmHeadChunkRows: 512,
});

const overrideListRuntime = createDopplerConfig({
  runtime: {
    inference: {
      largeWeights: {
        gpuResidentOverrides: ['model.language_model.embed_tokens.weight'],
      },
    },
  },
}).runtime;

assert.deepEqual(overrideListRuntime.inference.largeWeights.gpuResidentOverrides, [
  'model.language_model.embed_tokens.weight',
]);

console.log('runtime-large-weights-defaults-contract.test: ok');
