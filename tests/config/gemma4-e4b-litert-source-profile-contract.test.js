import assert from 'node:assert/strict';
import fs from 'node:fs';

import { resolveLiteRTPackageParsedArtifact } from '../../src/tooling/litert-package-runtime.js';
import { resolveDirectSourcePackageProfile } from '../../src/tooling/source-package-profiles.js';

const CONFIG_PATH = 'src/config/source-packages/litert/gemma-4-e4b-it.json';
const FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41];
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

  assert.equal(architecture?.numLayers, 42, `${label}: architecture layer count`);
  assert.equal(architecture?.hiddenSize, 2560, `${label}: hidden size`);
  assert.equal(architecture?.intermediateSize, 10240, `${label}: intermediate size`);
  assert.equal(architecture?.numKeyValueHeads, 2, `${label}: KV heads`);
  assert.equal(architecture?.numKvSharedLayers, 18, `${label}: KV shared layers`);
  assert.equal(textConfig?.num_hidden_layers, 42, `${label}: raw text layer count`);
  assert.equal(textConfig?.attention_k_eq_v, false, `${label}: attention K/V equality`);
  assert.equal(textConfig?.use_double_wide_mlp, false, `${label}: double-wide MLP`);
  assert.equal(rawLayerTypes.length, architecture.numLayers, `${label}: raw layer pattern length`);
  assert.deepEqual(
    rawLayerTypes
      .map((layerType, layerIndex) => (layerType === 'full_attention' ? layerIndex : null))
      .filter((layerIndex) => layerIndex != null),
    FULL_ATTENTION_LAYERS,
    `${label}: raw full-attention layers`
  );
  assert.equal(manifestLayerPattern?.type, 'every_n', `${label}: manifest layer pattern type`);
  assert.equal(manifestLayerPattern?.period, 6, `${label}: manifest layer pattern period`);
  assert.equal(manifestLayerPattern?.offset, 5, `${label}: manifest layer pattern offset`);
  assert.deepEqual(
    fullAttentionLayersFromEveryN(manifestLayerPattern, architecture.numLayers),
    FULL_ATTENTION_LAYERS,
    `${label}: manifest full-attention layers`
  );
  assert.deepEqual(profile.runtime?.tokenizer?.task?.eosTokens, [1, 106], `${label}: task EOS tokens`);
  assert.deepEqual(profile.runtime?.tokenizer?.litertlm?.eosTokens, [1, 106], `${label}: LiteRT-LM EOS tokens`);
}

function assertUnsupportedContract(profile, label) {
  const taskUnsupported = profile.package?.task?.unsupported;
  const litertlmUnsupported = profile.package?.litertlm?.unsupported;
  assert.equal(taskUnsupported?.code, UNSUPPORTED_CODE, `${label}: task unsupported code`);
  assert.match(taskUnsupported?.message ?? '', /no E4B direct-source parity receipt/i, `${label}: task unsupported message`);
  assert.equal(litertlmUnsupported?.code, UNSUPPORTED_CODE, `${label}: LiteRT-LM unsupported code`);
  assert.match(litertlmUnsupported?.recommendation ?? '', /graph contraction map/i, `${label}: LiteRT-LM recommendation`);
}

const sourceProfile = readJson(CONFIG_PATH);
const targetMatrix = readJson('models/gemma4-targets.json');
const e4bTarget = targetMatrix.targets.find((target) => target.targetId === 'gemma-4-e4b');

assertGemma4E4BContract(sourceProfile, 'source profile');
assertUnsupportedContract(sourceProfile, 'source profile');
assert.ok(e4bTarget, 'Gemma 4 E4B target must be listed');
assert.deepEqual(e4bTarget.sourcePackages, [
  {
    id: 'litert/gemma-4-e4b-it',
    status: 'blocked',
    blockerCode: UNSUPPORTED_CODE,
    reason: 'LiteRT .task and .litertlm package identity is known, but direct-source parsing fails closed until a parity receipt and graph contraction map exist.',
  },
]);
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
