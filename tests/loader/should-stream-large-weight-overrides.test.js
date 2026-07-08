import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import {
  requiresCpuF16ToF32MatmulMaterialization,
  resolveLargeWeightOverrides,
  shouldStreamLargeWeight,
} from '../../src/loader/manifest-config.js';

const fakeLocation = {
  shape: [262144, 1536],
  dtype: 'F16',
  role: 'embedding',
};

const fakeMatmulLocation = {
  shape: [1536, 896],
  dtype: 'F16',
  role: 'matmul',
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

  assert.equal(
    requiresCpuF16ToF32MatmulMaterialization(fakeMatmulLocation, { hasF16: false }, false),
    true,
    'F16 matmul weights require CPU F16->F32 materialization without shader-f16'
  );

  assert.equal(
    requiresCpuF16ToF32MatmulMaterialization(fakeMatmulLocation, { hasF16: true }, false),
    false,
    'F16 matmul weights stay F16 when shader-f16 is available'
  );

  assert.equal(
    requiresCpuF16ToF32MatmulMaterialization({ ...fakeMatmulLocation, dtype: 'BF16' }, { hasF16: false }, false),
    false,
    'BF16 conversion uses its own materialization path'
  );

  const testDir = path.dirname(fileURLToPath(import.meta.url));
  const loaderSource = fs.readFileSync(path.join(testDir, '../../src/loader/doppler-loader.js'), 'utf8');
  assert.match(
    loaderSource,
    /requiresCpuF16ToF32MatmulMaterialization\(location, this\.gpuCapabilities, this\.keepF32Weights\)/,
    'raw GPU streaming is gated by CPU F16->F32 materialization requirements'
  );
} finally {
  resetRuntimeConfig();
}

console.log('should-stream-large-weight-overrides.test: ok');
