import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);
const quickstartRegistry = JSON.parse(
  await fs.readFile(new URL('../../src/client/doppler-registry.json', import.meta.url), 'utf8')
);
const revocationRegistry = JSON.parse(
  await fs.readFile(new URL('../../src/config/revocation-registry.json', import.meta.url), 'utf8')
);
const { buildQuickstartRegistryPayload } = await import('../../tools/sync-quickstart-registry.js');

assert.deepEqual(
  quickstartRegistry,
  buildQuickstartRegistryPayload(catalog, revocationRegistry),
  'src/client/doppler-registry.json must be generated from models/catalog.json'
);

{
  const revoked = structuredClone(revocationRegistry);
  const modelId = quickstartRegistry.models[0].modelId;
  revoked.revocations.push({
    id: 'quickstart-filter-contract',
    state: 'revoked',
    issuedAtUtc: '2026-08-15T00:00:00.000Z',
    severity: 'policy',
    reason: 'Synthetic quickstart filtering contract.',
    targets: {
      logicalModelIds: [],
      modelIds: [modelId],
      sourceCheckpointIds: [],
      weightPackIds: [],
      manifestVariantIds: [],
      artifactVariantIds: [],
      adapterIds: [],
      adapterDigests: [],
    },
    replacements: {
      logicalModelIds: [],
      modelIds: [],
      sourceCheckpointIds: [],
      weightPackIds: [],
      manifestVariantIds: [],
      artifactVariantIds: [],
      adapterIds: [],
      adapterDigests: [],
    },
    evidencePaths: ['docs/goals.md'],
  });
  assert.equal(
    buildQuickstartRegistryPayload(catalog, revoked).models.some((entry) => entry.modelId === modelId),
    false,
    'revoked catalog models must be removed from generated quickstart resolution'
  );
}

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
    'classification',
    'vendorBenchmark',
    'benchmarkEvidence',
  ]) {
    assert.deepEqual(
      entry[field] ?? null,
      catalogEntry[field] ?? null,
      `${modelId}: ${field} must stay in sync with models/catalog.json`
    );
  }
}

console.log('quickstart-registry-sync.test: ok');
