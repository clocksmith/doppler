import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  MODEL_TYPE_TAXONOMY,
  resolveModelTypeCluster,
  validateCatalogClassifications,
  validateModelTypeTaxonomy,
} from '../../tools/lib/model-type-taxonomy.js';

const catalog = JSON.parse(fs.readFileSync('models/catalog.json', 'utf8'));

assert.deepEqual(validateModelTypeTaxonomy(), []);
assert.deepEqual(validateCatalogClassifications(catalog), []);

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
