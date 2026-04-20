import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KNOWN_MODELS } from '../../src/models/gemma4.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function assertGemmaDecodeLoop(decodeLoop, label) {
  assert.equal(decodeLoop?.batchSize, 8, `${label}: decode batch size`);
  assert.equal(decodeLoop?.stopCheckMode, 'batch', `${label}: stop check mode`);
  assert.equal(decodeLoop?.readbackInterval, 8, `${label}: readback interval`);
  assert.equal(decodeLoop?.readbackMode, 'overlapped', `${label}: readback mode`);
  assert.equal(decodeLoop?.ringTokens, 2, `${label}: token ring slots`);
  assert.equal(decodeLoop?.ringStop, 1, `${label}: stop ring slots`);
  assert.equal(decodeLoop?.ringStaging, 2, `${label}: staging ring slots`);
  assert.equal(decodeLoop?.disableCommandBatching, false, `${label}: command batching`);
}

function assertGemmaLargeWeights(largeWeights, label) {
  assert.deepEqual(
    largeWeights?.gpuResidentOverrides,
    ['model.language_model.embed_tokens.weight'],
    `${label}: embed_tokens.weight must be GPU-resident in the converted artifact contract`
  );
}

const baseConfig = await readJson('src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json');
const int4PleConfig = await readJson('src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json');
const baseManifest = await readJson('models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json');
const int4PleManifest = await readJson('models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json');

const modelProfiles = new Map(KNOWN_MODELS.map((entry) => [entry.modelId, entry.defaultRuntimeProfile]));
assert.equal(
  modelProfiles.get('gemma-4-e2b-it-q4k-ehf16-af32'),
  'profiles/throughput',
  'Gemma 4 base must rely on manifest-owned fast-path settings, not a model-scoped runtime preset'
);
assert.equal(
  modelProfiles.get('gemma-4-e2b-it-q4k-ehf16-af32-int4ple'),
  'profiles/throughput',
  'Gemma 4 INT4-PLE must rely on manifest-owned fast-path settings, not a model-scoped runtime preset'
);

for (const [label, config, manifest] of [
  ['gemma4-base', baseConfig, baseManifest],
  ['gemma4-int4ple', int4PleConfig, int4PleManifest],
]) {
  assertGemmaDecodeLoop(config.session?.decodeLoop, `${label} conversion config`);
  assertGemmaDecodeLoop(manifest.inference?.session?.decodeLoop, `${label} local manifest`);
  assertGemmaLargeWeights(config.largeWeights, `${label} conversion config`);
  assertGemmaLargeWeights(manifest.inference?.largeWeights, `${label} local manifest`);
  assert.equal(
    manifest.inference?.session?.perLayerInputs?.materialization,
    config.session?.perLayerInputs?.materialization,
    `${label}: local manifest must mirror conversion-config PLE materialization`
  );
}

assert.equal(
  baseConfig.session?.perLayerInputs?.materialization,
  'gpu_split_tables',
  'Gemma 4 base must carry the published fast PLE materialization in conversion config'
);
assert.equal(
  int4PleConfig.session?.perLayerInputs?.materialization,
  'range_backed',
  'Gemma 4 INT4-PLE keeps range-backed PLE materialization because INT4 PLE conversion rejects gpu_split_tables'
);

console.log('gemma4-e2b-runtime-profiles-contract.test: ok');
