import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { KB } from '../../src/config/schema/units.schema.js';

const defaults = createDopplerConfig().runtime.shared.bufferPool;

assert.equal(defaults.budget.hardFailOnBudgetExceeded, true, 'budget overruns must fail closed by default');

const overrideRuntime = createDopplerConfig({
  runtime: {
    shared: {
      bufferPool: {
        bucket: {
          largeBufferStepBytes: 128 * KB,
        },
        budget: {
          maxTotalBytes: 1234,
        },
      },
    },
  },
}).runtime;

assert.deepEqual(
  overrideRuntime.shared.bufferPool.bucket,
  { ...defaults.bucket, largeBufferStepBytes: 128 * KB },
  'a partial bucket override must keep every other bucket default'
);
assert.deepEqual(
  overrideRuntime.shared.bufferPool.budget,
  { ...defaults.budget, maxTotalBytes: 1234 },
  'a partial budget override must keep every other budget default'
);

console.log('buffer-pool-defaults-contract.test: ok');
