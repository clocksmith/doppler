import assert from 'node:assert/strict';
import fs from 'node:fs';

import { resolveDirectSourcePackageProfile } from '../../src/tooling/source-package-profiles.js';

const CONFIG_PATH = 'src/config/source-packages/litert/gemma-4-12b-it.json';
const FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41];
const QK_NORM_LAYERS = [2, 5, 9, 10, 11, 13, 15, 16, 17, 18, 23, 25, 30, 31, 33, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46];
const LAYER_SCALAR_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 46];
const UNSUPPORTED_CODE = 'gemma4-12b-litert-direct-source-unverified';

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
  const qkNormLayers = profile.runtime?.manifestInference?.attention?.queryKeyNormLayers;
  const qkNormWeightLayers = profile.runtime?.manifestInference?.attention?.queryKeyNormWeightLayers ?? [];

  assert.equal(architecture?.numLayers, 47, `${label}: architecture layer count`);
  assert.equal(textConfig?.num_hidden_layers, 47, `${label}: raw text layer count`);
  assert.equal(rawLayerTypes.length, architecture.numLayers, `${label}: raw layer pattern length`);
  assert.equal(manifestLayerTypes.length, architecture.numLayers, `${label}: manifest layer pattern length`);
  assert.deepEqual(fullAttentionLayers(rawLayerTypes), FULL_ATTENTION_LAYERS, `${label}: raw full-attention layers`);
  assert.deepEqual(fullAttentionLayers(manifestLayerTypes), FULL_ATTENTION_LAYERS, `${label}: manifest full-attention layers`);
  assert.equal(qkNormLayers, null, `${label}: Q/K norm op applies to all layers`);
  assert.deepEqual(qkNormWeightLayers, QK_NORM_LAYERS, `${label}: Q/K norm weight layers`);
}

function assertUnsupportedContract(profile, label) {
  const runtimeUnsupported = profile.runtime?.manifestInference?.unsupported;
  const packageUnsupported = profile.package?.litertlm?.unsupported;

  assert.equal(runtimeUnsupported?.code, UNSUPPORTED_CODE, `${label}: runtime unsupported code`);
  assert.match(
    runtimeUnsupported?.message ?? '',
    /47 repeated decoder-weight groups/,
    `${label}: runtime unsupported message`
  );
  assert.match(
    runtimeUnsupported?.recommendation ?? '',
    /graph contraction map/,
    `${label}: runtime unsupported recommendation`
  );
  assert.equal(packageUnsupported?.code, UNSUPPORTED_CODE, `${label}: package unsupported code`);
  assert.match(
    packageUnsupported?.message ?? '',
    /saturated logits/,
    `${label}: package unsupported message`
  );
}

const sourceProfile = readJson(CONFIG_PATH);
assertGemma412BLayerContract(sourceProfile, 'source profile');
assertUnsupportedContract(sourceProfile, 'source profile');
assert.equal(sourceProfile.package?.litertlm?.fixedInt4StorageEncoding, 'offset_binary');
assert.deepEqual(sourceProfile.package?.litertlm?.layerScalarLayers, LAYER_SCALAR_LAYERS);
assert.equal(sourceProfile.package?.litertlm?.missingLayerScalarValue, 1);

const resolvedProfile = resolveDirectSourcePackageProfile({
  sourceKind: 'litertlm',
  packageBasename: 'gemma-4-12B-it.litertlm',
});

assertGemma412BLayerContract(resolvedProfile, 'resolved source profile');
assertUnsupportedContract(resolvedProfile, 'resolved source profile');
assert.deepEqual(resolvedProfile.runtime.manifestInference.execution.prefill[1].layers, FULL_ATTENTION_LAYERS);
assert.equal(resolvedProfile.package?.litertlm?.fixedInt4StorageEncoding, 'offset_binary');
assert.deepEqual(resolvedProfile.package?.litertlm?.layerScalarLayers, LAYER_SCALAR_LAYERS);
assert.equal(resolvedProfile.package?.litertlm?.missingLayerScalarValue, 1);

console.log('gemma4-12b-litert-source-profile-contract.test: ok');
