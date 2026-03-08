import assert from 'node:assert/strict';

import { chooseDefined } from '../../src/config/merge-helpers.js';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

assert.equal(
  chooseDefined(undefined, 'manifest-path'),
  'manifest-path'
);

assert.equal(
  chooseDefined(null, 'manifest-path'),
  null
);

const nullKernelPathConfig = createDopplerConfig({
  runtime: {
    inference: {
      kernelPath: null,
    },
  },
});

assert.equal(
  Object.prototype.hasOwnProperty.call(nullKernelPathConfig.runtime.inference, 'kernelPath'),
  true
);
assert.equal(nullKernelPathConfig.runtime.inference.kernelPath, null);

console.log('doppler-schema-kernel-path-null.test: ok');
