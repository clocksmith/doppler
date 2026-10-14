import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);
const quickstartRegistry = JSON.parse(
  await fs.readFile(new URL('../../src/config/quickstart-registry.json', import.meta.url), 'utf8')
);
const revocationRegistry = JSON.parse(
  await fs.readFile(new URL('../../src/config/revocation-registry.json', import.meta.url), 'utf8')
);
const { buildQuickstartRegistryPayload } = await import('../../tools/sync-quickstart-registry.js');

assert.deepEqual(
  quickstartRegistry,
  buildQuickstartRegistryPayload(catalog, revocationRegistry),
  'src/config/quickstart-registry.json must be generated from models/catalog.json'
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
      weightCapsuleIds: [],
      manifestVariantIds: [],
      artifactVariantIds: [],
      adapterIds: [],
      adapterDigests: [],
    },
    replacements: {
      logicalModelIds: [],
      modelIds: [],
      sourceCheckpointIds: [],
      weightCapsuleIds: [],
      manifestVariantIds: [],
      artifactVariantIds: [],
      adapterIds: [],
      adapterDigests: [],
    },
    evidencePaths: ['GOALS.md'],
  });
  assert.equal(
    buildQuickstartRegistryPayload(catalog, revoked).models.some((entry) => entry.modelId === modelId),
    false,
    'revoked catalog models must be removed from generated quickstart resolution'
  );
}

console.log('quickstart-registry-sync.test: ok');
