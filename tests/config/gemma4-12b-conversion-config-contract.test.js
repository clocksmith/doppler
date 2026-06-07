import assert from 'node:assert/strict';
import fs from 'node:fs';

const MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af32';
const CONFIG_PATH = `src/config/conversion/gemma4/${MODEL_ID}.json`;
const FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47];
const PROJECTION_OPS = new Set(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function phaseStepGroups(execution, phase) {
  const phaseValue = execution?.[phase];
  if (!Array.isArray(phaseValue)) {
    return [];
  }
  if (phase === 'prefill' && phaseValue.every((entry) => !Array.isArray(entry) && Array.isArray(entry?.steps))) {
    return phaseValue.map((entry) => entry.steps);
  }
  return [phaseValue];
}

function assertProjectionPath(execution, phase, expectedKernelRef, label) {
  const groups = phaseStepGroups(execution, phase);
  assert.ok(groups.length > 0, `${label}: ${phase} steps must exist`);
  for (const steps of groups) {
    for (const step of steps) {
      if (!PROJECTION_OPS.has(step[0])) {
        continue;
      }
      assert.equal(step[1], expectedKernelRef, `${label}: ${phase} ${step[0]} must use ${expectedKernelRef}`);
    }
  }
}

const conversionConfig = readJson(CONFIG_PATH);
const layerTypes = conversionConfig.inference?.layerPattern?.layerTypes ?? [];
const actualFullAttentionLayers = layerTypes
  .map((kind, index) => (kind === 'full_attention' ? index : null))
  .filter((index) => index != null);

assert.equal(conversionConfig.modelType, 'gemma4');
assert.equal(conversionConfig.output?.modelBaseId, MODEL_ID);
assert.equal(conversionConfig.output?.textOnly, true);
assert.equal(conversionConfig.manifest?.artifactIdentity?.sourceCheckpointId, 'google/gemma-4-12B-it');
assert.equal(conversionConfig.manifest?.artifactIdentity?.sourceRevision, '5926caa4ec0cac5cbfadaf4077420520de1d5205');
assert.equal(conversionConfig.manifest?.artifactIdentity?.manifestVariantId, `${MODEL_ID}-mv-q4-fused-v1`);

assert.equal(conversionConfig.manifest?.visionConfig?.vision_architecture, 'gemma4');
assert.equal(conversionConfig.manifest?.visionConfig?.depth, 0);
assert.equal(conversionConfig.manifest?.visionConfig?.hidden_size, 3840);
assert.equal(conversionConfig.manifest?.audioConfig?.audio_architecture, 'gemma4');
assert.equal(conversionConfig.manifest?.audioConfig?.num_hidden_layers, 0);
assert.equal(conversionConfig.manifest?.audioConfig?.hidden_size, 640);

assert.equal(conversionConfig.inference?.attention?.slidingWindow, 1024);
assert.equal(conversionConfig.inference?.attention?.queryPreAttnScalar, 1);
assert.equal(conversionConfig.inference?.ffn?.useDoubleWideMlp, false);
assert.equal(conversionConfig.inference?.output?.embeddingVocabSize, 262144);
assert.equal(conversionConfig.session?.decodeLoop?.batchSize, 1);
assert.equal(conversionConfig.session?.decodeLoop?.readbackInterval, 1);
assert.equal(layerTypes.length, 48);
assert.deepEqual(actualFullAttentionLayers, FULL_ATTENTION_LAYERS);

assert.deepEqual(conversionConfig.execution?.prefill?.[1]?.layers, FULL_ATTENTION_LAYERS);
assertProjectionPath(conversionConfig.execution, 'decode', 'q4_decode_gemv', 'conversion config');
assertProjectionPath(conversionConfig.execution, 'prefill', 'q4_widetile', 'conversion config');

console.log('gemma4-12b-conversion-config-contract.test: ok');
