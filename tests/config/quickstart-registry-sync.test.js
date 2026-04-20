import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);
const quickstartRegistry = JSON.parse(
  await fs.readFile(new URL('../../src/client/doppler-registry.json', import.meta.url), 'utf8')
);
const { buildQuickstartRegistryPayload } = await import('../../tools/sync-quickstart-registry.js');

assert.deepEqual(
  quickstartRegistry,
  buildQuickstartRegistryPayload(catalog),
  'src/client/doppler-registry.json must be generated from models/catalog.json'
);

const catalogByModelId = new Map(
  (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => entry && typeof entry.modelId === 'string')
    .map((entry) => [entry.modelId, entry])
);

for (const entry of Array.isArray(quickstartRegistry?.models) ? quickstartRegistry.models : []) {
  const modelId = typeof entry?.modelId === 'string' ? entry.modelId : '';
  assert.ok(modelId, 'quickstart registry entries must define modelId');

  const catalogEntry = catalogByModelId.get(modelId);
  assert.ok(catalogEntry, `quickstart registry entry "${modelId}" must exist in models/catalog.json`);
  assert.equal(catalogEntry.quickstart, true, `${modelId}: quickstart registry entries must set quickstart=true in models/catalog.json`);

  assert.deepEqual(
    entry.aliases ?? [],
    catalogEntry.aliases ?? [],
    `${modelId}: aliases must stay in sync with models/catalog.json`
  );
  assert.deepEqual(
    entry.modes ?? [],
    catalogEntry.modes ?? [],
    `${modelId}: modes must stay in sync with models/catalog.json`
  );
  assert.deepEqual(
    entry.hf ?? null,
    catalogEntry.hf ?? null,
    `${modelId}: hf metadata must stay in sync with models/catalog.json`
  );
  for (const field of [
    'sourceCheckpointId',
    'weightPackId',
    'manifestVariantId',
    'artifactCompleteness',
    'runtimePromotionState',
    'weightsRefAllowed',
  ]) {
    assert.deepEqual(
      entry[field] ?? null,
      catalogEntry[field] ?? null,
      `${modelId}: ${field} must stay in sync with models/catalog.json`
    );
  }
}

console.log('quickstart-registry-sync.test: ok');
