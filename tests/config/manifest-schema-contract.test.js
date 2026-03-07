import assert from 'node:assert/strict';

import {
  DEFAULT_MANIFEST_INFERENCE,
  validateManifestInference,
} from '../../src/config/schema/index.js';

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

{
  const manifest = {
    modelId: 'valid-manifest',
    modelType: 'transformer',
    inference: clone(DEFAULT_MANIFEST_INFERENCE),
  };
  validateManifestInference(manifest);
}

{
  const manifest = {
    modelId: 'invalid-manifest',
    modelType: 'transformer',
    inference: clone(DEFAULT_MANIFEST_INFERENCE),
  };
  delete manifest.inference.output.finalLogitSoftcapping;
  assert.throws(
    () => validateManifestInference(manifest),
    /Manifest "invalid-manifest" has incomplete inference config/
  );
}

{
  const diffusionManifest = {
    modelId: 'diffusion-manifest',
    modelType: 'diffusion',
    inference: {
      presetId: 'diffusion',
    },
  };
  validateManifestInference(diffusionManifest);
}

console.log('manifest-schema-contract.test: ok');
