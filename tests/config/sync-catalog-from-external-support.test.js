import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { buildCatalogFromExternalRegistry } from '../../tools/sync-catalog-from-external-support.js';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-catalog-from-external-'));

try {
  const registryPath = path.join(root, 'DOPPLER_SUPPORT_REGISTRY.json');
  writeFileSync(registryPath, JSON.stringify({
    version: 1,
    lifecycleSchemaVersion: 1,
    updatedAt: '2026-03-11',
    generatedAt: '2026-03-11T12:00:00.000Z',
    models: [
      {
        modelId: 'demo-model',
        modes: ['run'],
        artifact: {
          format: 'rdrr',
        },
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'abc123abc123abc123abc123abc123abc123abcd',
          path: 'models/demo-model',
        },
        lifecycle: {
          availability: {
            hf: true,
          },
        },
        external: {
          rdrrModelId: 'demo-model',
        },
      },
    ],
    uncatalogedRdrrVariants: [],
  }, null, 2), 'utf8');

  const serialized = await buildCatalogFromExternalRegistry({
    externalRegistry: registryPath,
    catalogFile: path.join(root, 'catalog.json'),
  });
  const payload = JSON.parse(serialized);
  assert.equal(payload.models.length, 1);
  assert.equal(payload.models[0].modelId, 'demo-model');
  assert.equal(Object.prototype.hasOwnProperty.call(payload.models[0], 'external'), false);
}
finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('sync-catalog-from-external-support.test: ok');
