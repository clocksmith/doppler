import assert from 'node:assert/strict';

import {
  DEFAULT_MANIFEST_INFERENCE,
  validateManifestInference,
} from '../../src/config/schema/index.js';

const { parseModelConfig } = await import('../../src/inference/pipelines/text/config.js');

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const manifest = {
  modelId: 'unsupported-manifest-test',
  modelType: 'transformer',
  architecture: {},
  inference: {
    ...clone(DEFAULT_MANIFEST_INFERENCE),
    unsupported: {
      code: 'fixture-unverified-contract',
      message: 'The fixture graph contract is intentionally blocked.',
      recommendation: 'Use a manifest with verified graph metadata.',
    },
  },
};

assert.throws(
  () => parseModelConfig(manifest, null),
  /unsupported-manifest-test.*fixture-unverified-contract.*intentionally blocked.*verified graph metadata/
);

assert.throws(
  () => validateManifestInference(manifest),
  /unsupported-manifest-test.*fixture-unverified-contract.*intentionally blocked.*verified graph metadata/
);

console.log('manifest-unsupported-contract.test: ok');
