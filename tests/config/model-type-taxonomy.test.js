import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  MODEL_TYPE_TAXONOMY,
  buildModelTypeClusters,
  resolveModelTypeCluster,
  validateCatalogClassifications,
  validateModelTypeTaxonomy,
} from '../../tools/lib/model-type-taxonomy.js';

const catalog = JSON.parse(fs.readFileSync('models/catalog.json', 'utf8'));

assert.deepEqual(validateModelTypeTaxonomy(), []);
assert.deepEqual(validateCatalogClassifications(catalog), []);

const clusters = buildModelTypeClusters(catalog.models);
assert.deepEqual(
  Object.fromEntries(clusters.map((cluster) => [cluster.id, cluster.models.length])),
  {
    'text-generators': 14,
    'multimodal-generators': 3,
    'diffusion-language-models': 1,
    'translation-specialists': 2,
    'language-embedders': 2,
    rerankers: 2,
    'protein-encoders': 3,
    'nucleotide-encoders': 1,
  }
);

const byModelId = new Map(catalog.models.map((model) => [model.modelId, model]));
const expectedClusters = new Map([
  ['gemma-3-270m-it-q4k-ehf16-af32', 'text-generators'],
  ['gemma-4-e2b-it-q4k-ehf16-af32', 'multimodal-generators'],
  ['diffusiongemma-26b-a4b-it-q4k-ehf16-af16', 'diffusion-language-models'],
  ['translategemma-4b-it-q4k-ehf16-af32', 'translation-specialists'],
  ['qwen-3-embedding-0-6b-q4k-ehf16-af32', 'language-embedders'],
  ['qwen-3-reranker-0-6b-q4k-ehf16-af32', 'rerankers'],
  ['esm2-t12-35m-ur50d-f32-af32', 'protein-encoders'],
  ['nucleotide-transformer-v2-50m-f32-af32', 'nucleotide-encoders'],
]);
for (const [modelId, clusterId] of expectedClusters) {
  assert.equal(resolveModelTypeCluster(byModelId.get(modelId).classification).id, clusterId, modelId);
}

for (const modelId of [
  'gemma-3-270m-it-q4k-ehf16-af32',
  'gemma-3-270m-it-f16-af32',
  'gemma-3-1b-it-q4k-ehf16-af32',
]) {
  assert.deepEqual(byModelId.get(modelId).modes, ['text'], `${modelId} is a text-only RDRR artifact`);
  assert.deepEqual(byModelId.get(modelId).classification.inputs, ['text']);
}

const ambiguousTaxonomy = structuredClone(MODEL_TYPE_TAXONOMY);
ambiguousTaxonomy.clusters.push({
  ...structuredClone(ambiguousTaxonomy.clusters[0]),
  id: 'duplicate-text-generators',
});
assert.throws(
  () => resolveModelTypeCluster(byModelId.get('gemma-3-270m-it-q4k-ehf16-af32').classification, ambiguousTaxonomy),
  /must resolve to exactly one model type cluster/
);

console.log('model-type-taxonomy.test: ok');
