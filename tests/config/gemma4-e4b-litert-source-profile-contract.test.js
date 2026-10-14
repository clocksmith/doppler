import assert from 'node:assert/strict';
import fs from 'node:fs';

import { resolveLiteRTPackageParsedArtifact } from '../../src/tooling/litert-package-runtime.js';
import { resolveDirectSourcePackageProfile } from '../../src/tooling/source-package-profiles.js';

const CONFIG_PATH = 'src/config/source-packages/litert/gemma-4-e4b-it.json';
const UNSUPPORTED_CODE = 'gemma4-e4b-litert-direct-source-unverified';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function fullAttentionLayersFromEveryN(layerPattern, numLayers) {
  const layers = [];
  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    if (((layerIndex - layerPattern.offset) % layerPattern.period) === 0) {
      layers.push(layerIndex);
    }
  }
  return layers;
}

function assertGemma4E4BContract(profile, label) {
  const architecture = profile.runtime?.architecture;
  const textConfig = profile.runtime?.rawConfig?.text_config;
  const rawLayerTypes = textConfig?.layer_types ?? [];
  const manifestLayerPattern = profile.runtime?.manifestInference?.layerPattern;

  assert.equal(textConfig?.num_hidden_layers, architecture?.numLayers, `${label}: raw text layer count`);
  assert.equal(rawLayerTypes.length, architecture.numLayers, `${label}: raw layer pattern length`);
  assert.deepEqual(
    fullAttentionLayersFromEveryN(manifestLayerPattern, architecture.numLayers),
    rawLayerTypes
      .map((layerType, layerIndex) => (layerType === 'full_attention' ? layerIndex : null))
      .filter((layerIndex) => layerIndex != null),
    `${label}: manifest full-attention layers must match the raw source layer types`
  );
  assert.deepEqual(
    profile.runtime?.tokenizer?.litertlm?.eosTokens,
    profile.runtime?.tokenizer?.task?.eosTokens,
    `${label}: LiteRT-LM and task EOS tokens must agree`
  );
}

function assertUnsupportedContract(profile, label) {
  assert.equal(profile.package?.task?.unsupported?.code, UNSUPPORTED_CODE, `${label}: task unsupported code`);
  assert.equal(profile.package?.litertlm?.unsupported?.code, UNSUPPORTED_CODE, `${label}: LiteRT-LM unsupported code`);
}

const sourceProfile = readJson(CONFIG_PATH);
const targetMatrix = readJson('models/gemma4-targets.json');
const e4bTarget = targetMatrix.targets.find((target) => target.targetId === 'gemma-4-e4b');

assertGemma4E4BContract(sourceProfile, 'source profile');
assertUnsupportedContract(sourceProfile, 'source profile');
assert.ok(e4bTarget, 'Gemma 4 E4B target must be listed');
const sourcePackage = e4bTarget.sourcePackages?.find((entry) => entry.id === 'litert/gemma-4-e4b-it');
assert.ok(sourcePackage, 'E4B target must list its LiteRT source package');
assert.equal(sourcePackage.status, 'blocked', 'E4B LiteRT source package must stay blocked until parity evidence exists');
assert.equal(sourcePackage.blockerCode, UNSUPPORTED_CODE);
assert.ok(!e4bTarget.missing.includes('source package profile'), 'E4B target must not claim the source package profile is missing');

for (const [sourceKind, packageBasename] of [
  ['litert-task', 'gemma-4-E4B-it-web.task'],
  ['litertlm', 'gemma-4-E4B-it.litertlm'],
  ['litertlm', 'gemma-4-E4B-it-web.litertlm'],
]) {
  const resolvedProfile = resolveDirectSourcePackageProfile({ sourceKind, packageBasename });
  assertGemma4E4BContract(resolvedProfile, `${sourceKind} resolved profile`);
  assertUnsupportedContract(resolvedProfile, `${sourceKind} resolved profile`);

  await assert.rejects(
    resolveLiteRTPackageParsedArtifact({
      sourceKind,
      sourcePathForModelId: `/fixture/${packageBasename}`,
      source: {
        name: packageBasename,
        size: 0,
        async readRange() {
          return new ArrayBuffer(0);
        },
      },
    }),
    new RegExp(`${UNSUPPORTED_CODE}.*E4B`, 'i'),
    `${sourceKind} ${packageBasename} must fail closed before direct-source parsing`
  );
}

console.log('gemma4-e4b-litert-source-profile-contract.test: ok');
