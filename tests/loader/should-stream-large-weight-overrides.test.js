import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import {
  resolveLargeWeightOverrides,
  shouldStreamLargeWeight,
} from '../../src/loader/manifest-config.js';

const fakeLocation = {
  shape: [262144, 1536],
  dtype: 'F16',
  role: 'embedding',
};

try {
  resetRuntimeConfig();
  assert.deepEqual(
    resolveLargeWeightOverrides(['manifest.weight'], null),
    ['manifest.weight'],
    'manifest overrides apply when runtime override is absent'
  );
  assert.deepEqual(
    resolveLargeWeightOverrides(['manifest.weight'], []),
    [],
    'empty runtime override explicitly replaces manifest overrides'
  );
  assert.deepEqual(
    resolveLargeWeightOverrides(['manifest.weight'], ['runtime.weight']),
    ['runtime.weight'],
    'runtime override list replaces manifest overrides'
  );

  assert.equal(
    shouldStreamLargeWeight(
      'model.language_model.embed_tokens.weight',
      fakeLocation,
      'Embedding'
    ),
    false,
    'no GPU device + null overrides falls through and returns false (no streaming)'
  );

  setRuntimeConfig({
    inference: {
      largeWeights: {
        gpuResidentOverrides: ['model.language_model.embed_tokens.weight'],
      },
    },
  });
  assert.equal(
    shouldStreamLargeWeight(
      'model.language_model.embed_tokens.weight',
      fakeLocation,
      'Embedding'
    ),
    false,
    'matching name in override list returns false (force GPU-resident)'
  );

  assert.equal(
    shouldStreamLargeWeight(
      'model.language_model.lm_head.weight',
      fakeLocation,
      'LM head'
    ),
    false,
    'unrelated name not in override list passes through unchanged'
  );

  setRuntimeConfig({
    inference: {
      largeWeights: {
        gpuResidentOverrides: null,
      },
    },
  });
  assert.equal(
    shouldStreamLargeWeight(
      'model.language_model.embed_tokens.weight',
      fakeLocation,
      'Embedding'
    ),
    false,
    'null overrides is valid (default) and does not crash'
  );

  setRuntimeConfig({
    inference: {
      largeWeights: {
        gpuResidentOverrides: [],
      },
    },
  });
  assert.equal(
    shouldStreamLargeWeight(
      'model.language_model.embed_tokens.weight',
      fakeLocation,
      'Embedding'
    ),
    false,
    'empty overrides array is valid and matches nothing'
  );
} finally {
  resetRuntimeConfig();
}

console.log('should-stream-large-weight-overrides.test: ok');
