import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const targets = JSON.parse(readFileSync('models/gemma4-targets.json', 'utf8'));
const catalog = JSON.parse(readFileSync('models/catalog.json', 'utf8'));
const genericMoeConfig = JSON.parse(
  readFileSync('src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json', 'utf8')
);

const OFFICIAL_26B_A4B_SOURCE = 'google/gemma-4-26B-A4B-it';
const DIFFUSIONGEMMA_26B_A4B_MODEL_ID = 'diffusiongemma-26b-a4b-it-q4k-ehf16-af16';

const OFFICIAL_26B_A4B_MOE_ROUTE = Object.freeze({
  numExperts: 128,
  topKExperts: 8,
});

const targetsById = new Map(targets.targets.map((target) => [target.targetId, target]));
const catalogById = new Map(catalog.models.map((model) => [model.modelId, model]));

function getSourceCheckpointId(model) {
  return model.sourceCheckpointId ?? model.artifact?.sourceCheckpointId ?? null;
}

const target = targetsById.get('gemma-4-26b-a4b');
assert.ok(target, 'Gemma 4 target matrix must include official 26B A4B');
assert.equal(target.officialName, 'Gemma 4 26B A4B');
assert.equal(target.dopplerStatus, 'gap', 'official 26B A4B must remain a gap until a real lane exists');
assert.deepEqual(target.surfaceStatus, {
  browser: 'unsupported',
  electron: 'unsupported',
  node: 'unsupported',
});
assert.deepEqual(target.currentLanes, [], 'official 26B A4B gap must not list current lanes');

for (const missing of [
  'conversion config',
  'catalog model',
  'MoE runtime receipt',
  'benchmark receipt',
  'mtp lane',
]) {
  assert.ok(target.missing.includes(missing), `official 26B A4B gap must list missing ${missing}`);
}

assert.ok(
  target.relatedButNotEquivalent?.includes(DIFFUSIONGEMMA_26B_A4B_MODEL_ID),
  'DiffusionGemma 26B A4B must be called out as related but not equivalent'
);

for (const model of catalog.models) {
  assert.notEqual(
    model.modelId,
    'gemma-4-26b-a4b',
    'official 26B A4B must not be cataloged under a generic target ID'
  );
  assert.notEqual(
    getSourceCheckpointId(model),
    OFFICIAL_26B_A4B_SOURCE,
    'official 26B A4B source checkpoint must not appear in catalog before support is real'
  );
}

const diffusionGemma = catalogById.get(DIFFUSIONGEMMA_26B_A4B_MODEL_ID);
assert.ok(diffusionGemma, 'DiffusionGemma related model must remain cataloged separately');
assert.equal(diffusionGemma.family, 'diffusiongemma');
assert.equal(getSourceCheckpointId(diffusionGemma), 'google/diffusiongemma-26B-A4B-it');
assert.notEqual(getSourceCheckpointId(diffusionGemma), OFFICIAL_26B_A4B_SOURCE);

assert.equal(genericMoeConfig.output?.modelBaseId, 'gemma-4-moe-q4k-ehf16-af32');
assert.notEqual(
  genericMoeConfig.output?.modelBaseId,
  'gemma-4-26b-a4b-it-text-q4k-ehf16-af32',
  'generic MoE config must not masquerade as official 26B A4B'
);
assert.equal(genericMoeConfig.inference?.moe?.kernelProfileId, 'mixtral-moe-v1');
assert.equal(genericMoeConfig.inference?.moe?.tensorPattern, 'mixtral');
assert.equal(genericMoeConfig.inference?.moe?.numExperts, 8);
assert.equal(genericMoeConfig.inference?.moe?.topK, 2);
assert.notEqual(genericMoeConfig.inference?.moe?.numExperts, OFFICIAL_26B_A4B_MOE_ROUTE.numExperts);
assert.notEqual(genericMoeConfig.inference?.moe?.topK, OFFICIAL_26B_A4B_MOE_ROUTE.topKExperts);

console.log('gemma4-26b-a4b-target-contract.test: ok');
