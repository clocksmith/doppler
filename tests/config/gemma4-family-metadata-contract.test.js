import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { KNOWN_MODELS, resolveHfBaseUrl, resolveModel } from '../../src/models/gemma4.js';

const catalog = JSON.parse(readFileSync('models/catalog.json', 'utf8'));

const verifiedHostedGemma4Models = catalog.models
  .filter((model) => model.family === 'gemma4')
  .filter((model) => model.lifecycle?.availability?.hf === true)
  .filter((model) => model.lifecycle?.status?.tested === 'verified')
  .sort((left, right) => left.modelId.localeCompare(right.modelId));

const knownById = new Map(KNOWN_MODELS.map((entry) => [entry.modelId, entry]));

for (const catalogModel of verifiedHostedGemma4Models) {
  const entry = knownById.get(catalogModel.modelId);
  assert.ok(entry, `models/gemma4 must expose verified hosted Gemma 4 model ${catalogModel.modelId}`);
  assert.equal(entry.label, catalogModel.label, `${catalogModel.modelId}: label must match catalog`);
  assert.equal(entry.hfPath, catalogModel.hf.path, `${catalogModel.modelId}: HF path must match catalog`);
  assert.deepEqual([...entry.modes].sort(), [...catalogModel.modes].sort(), `${catalogModel.modelId}: modes must match catalog`);
  assert.equal(resolveModel(catalogModel.modelId), entry, `${catalogModel.modelId}: resolveModel must return the known entry`);
  assert.equal(
    resolveHfBaseUrl(catalogModel.modelId, 'main'),
    `https://huggingface.co/clocksmith/rdrr/resolve/main/${catalogModel.hf.path}`,
    `${catalogModel.modelId}: resolveHfBaseUrl must point at the hosted artifact path`
  );
}

for (const entry of KNOWN_MODELS) {
  assert.ok(
    catalog.models.some((model) => model.modelId === entry.modelId),
    `${entry.modelId}: models/gemma4 entry must have a catalog model`
  );
}

console.log('gemma4-family-metadata-contract.test: ok');
