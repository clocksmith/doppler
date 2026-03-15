import assert from 'node:assert/strict';

import {
  buildQuickstartModelBaseUrl,
  resolveQuickstartModel,
} from '../../src/client/doppler-registry.js';
import { listQuickstartModels } from '../../src/client/doppler-registry.js';
import { MODEL_REQUIREMENTS } from '../../src/storage/preflight.js';
import {
  getCDNBaseUrl,
  listQuickStartModels,
} from '../../src/storage/quickstart-downloader.js';

const registryModels = await listQuickstartModels();
const registryIds = registryModels.map((entry) => entry.modelId).sort();
const storageIds = listQuickStartModels().map((entry) => entry.modelId).sort();

assert.deepEqual(storageIds, registryIds);

for (const modelId of storageIds) {
  assert.ok(MODEL_REQUIREMENTS[modelId], `${modelId}: quickstart downloader requires preflight requirements`);
  const storageEntry = listQuickStartModels().find((entry) => entry.modelId === modelId);
  const registryEntry = await resolveQuickstartModel(modelId);
  assert.equal(storageEntry?.baseUrl ?? null, null, `${modelId}: quickstart downloader must not assume same-origin model hosting`);
  assert.deepEqual(storageEntry?.hf ?? null, registryEntry.hf, `${modelId}: storage quickstart source must match the generated HF registry`);
  assert.equal(
    buildQuickstartModelBaseUrl(storageEntry, { cdnBasePath: getCDNBaseUrl() }),
    buildQuickstartModelBaseUrl(registryEntry, { cdnBasePath: getCDNBaseUrl() }),
    `${modelId}: storage quickstart URL must match the generated HF registry`
  );
}

console.log('quickstart-storage-registry-sync.test: ok');
