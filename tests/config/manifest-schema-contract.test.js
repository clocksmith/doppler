import assert from 'node:assert/strict';

import {
  DEFAULT_MANIFEST_INFERENCE,
  DEFAULT_PRESET_INFERENCE_CONFIG,
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

assert.equal(DEFAULT_MANIFEST_INFERENCE.rope.ropeScalingType, null);
assert.equal(DEFAULT_MANIFEST_INFERENCE.rope.yarnBetaFast, null);
assert.equal(DEFAULT_MANIFEST_INFERENCE.rope.yarnBetaSlow, null);
assert.equal(DEFAULT_MANIFEST_INFERENCE.rope.yarnOriginalMaxPos, null);

assert.equal(DEFAULT_PRESET_INFERENCE_CONFIG.rope.ropeScalingType, null);
assert.equal(DEFAULT_PRESET_INFERENCE_CONFIG.rope.yarnBetaFast, null);
assert.equal(DEFAULT_PRESET_INFERENCE_CONFIG.rope.yarnBetaSlow, null);
assert.equal(DEFAULT_PRESET_INFERENCE_CONFIG.rope.yarnOriginalMaxPos, null);

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

{
  const energyManifest = {
    modelId: 'energy-manifest',
    modelType: 'energy',
    inference: {
      presetId: 'energy',
    },
  };
  validateManifestInference(energyManifest);
}

console.log('manifest-schema-contract.test: ok');
