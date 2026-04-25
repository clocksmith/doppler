import assert from 'node:assert/strict';

import { applyModelBatchingRuntimeDefaults } from '../../src/inference/pipelines/text/model-load.js';

const manifest = {
  modelId: 'large-weights-runtime-override-fixture',
  inference: {
    session: {
      decodeLoop: {
        batchSize: 1,
        stopCheckMode: 'batch',
        readbackInterval: 1,
        readbackMode: 'sequential',
      },
    },
    largeWeights: {
      gpuResidentOverrides: ['manifest.weight'],
    },
  },
};

{
  const runtimeConfig = {
    inference: {
      session: {},
      batching: {},
      largeWeights: {
        gpuResidentOverrides: null,
      },
    },
  };
  const merged = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, {});
  assert.deepEqual(
    merged.inference.largeWeights.gpuResidentOverrides,
    ['manifest.weight'],
    'manifest largeWeights are promoted when runtime does not explicitly override'
  );
}

{
  const runtimeConfig = {
    inference: {
      session: {},
      batching: {},
      largeWeights: {
        gpuResidentOverrides: [],
      },
    },
  };
  const merged = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, {});
  assert.deepEqual(
    merged.inference.largeWeights.gpuResidentOverrides,
    [],
    'empty runtime largeWeights override suppresses manifest gpu-resident overrides'
  );
}

console.log('model-load-large-weights-runtime-override.test: ok');
