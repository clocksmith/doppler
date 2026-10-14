import assert from 'node:assert/strict';
import fs from 'node:fs';

import { resolveDirectSourcePackageProfile } from '../../src/tooling/source-package-profiles.js';

const CONFIG_PATH = 'src/config/source-packages/litert/gemma-4-12b-it.json';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function fullAttentionLayers(layerTypes) {
  return layerTypes
    .map((layerType, layerIndex) => (layerType === 'full_attention' ? layerIndex : null))
    .filter((layerIndex) => layerIndex != null);
}

function assertGemma412BLayerContract(profile, label) {
  const architecture = profile.runtime?.architecture;
  const textConfig = profile.runtime?.rawConfig?.text_config;
  const rawLayerTypes = textConfig?.layer_types ?? [];
  const manifestLayerTypes = profile.runtime?.manifestInference?.layerPattern?.layerTypes ?? [];

  assert.equal(textConfig?.num_hidden_layers, architecture?.numLayers, `${label}: raw text layer count`);
  assert.equal(rawLayerTypes.length, architecture.numLayers, `${label}: raw layer pattern length`);
  assert.equal(manifestLayerTypes.length, architecture.numLayers, `${label}: manifest layer pattern length`);
  assert.deepEqual(
    fullAttentionLayers(manifestLayerTypes),
    fullAttentionLayers(rawLayerTypes),
    `${label}: manifest full-attention layers must match the raw source layer types`
  );
}

function assertUnsupportedContract(profile, label) {
  const runtimeUnsupported = profile.runtime?.manifestInference?.unsupported;
  const packageUnsupported = profile.package?.litertlm?.unsupported;

  assert.ok(runtimeUnsupported?.code, `${label}: runtime must stay marked unsupported`);
  assert.equal(packageUnsupported?.code, runtimeUnsupported.code, `${label}: package unsupported code`);
}

const sourceProfile = readJson(CONFIG_PATH);
assertGemma412BLayerContract(sourceProfile, 'source profile');
assertUnsupportedContract(sourceProfile, 'source profile');

const resolvedProfile = resolveDirectSourcePackageProfile({
  sourceKind: 'litertlm',
  packageBasename: 'gemma-4-12B-it.litertlm',
});

assertGemma412BLayerContract(resolvedProfile, 'resolved source profile');
assertUnsupportedContract(resolvedProfile, 'resolved source profile');
assert.deepEqual(
  resolvedProfile.runtime.manifestInference.execution.prefill[1].layers,
  fullAttentionLayers(resolvedProfile.runtime.manifestInference.layerPattern.layerTypes),
  'resolved source profile: full-attention prefill section must cover exactly the full-attention layers'
);

console.log('gemma4-12b-litert-source-profile-contract.test: ok');
