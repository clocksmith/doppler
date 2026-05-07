import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { KB } from '../../src/config/schema/units.schema.js';

const defaults = createDopplerConfig().runtime.shared.bufferPool.bucket;

assert.equal(defaults.minBucketSizeBytes, 256);
assert.equal(defaults.largeBufferThresholdBytes, 64 * KB);
assert.equal(defaults.largeBufferStepBytes, 64 * KB);

const overrideRuntime = createDopplerConfig({
  runtime: {
    shared: {
      bufferPool: {
        bucket: {
          largeBufferStepBytes: 128 * KB,
        },
      },
    },
  },
}).runtime;

assert.equal(overrideRuntime.shared.bufferPool.bucket.minBucketSizeBytes, 256);
assert.equal(overrideRuntime.shared.bufferPool.bucket.largeBufferThresholdBytes, 64 * KB);
assert.equal(overrideRuntime.shared.bufferPool.bucket.largeBufferStepBytes, 128 * KB);
