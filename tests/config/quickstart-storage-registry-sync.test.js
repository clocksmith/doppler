import assert from 'node:assert/strict';

import { listQuickstartModels } from '../../src/client/doppler-registry.js';
import { MODEL_REQUIREMENTS } from '../../src/storage/preflight.js';
import { listQuickStartModels } from '../../src/storage/quickstart-downloader.js';

const registryModels = await listQuickstartModels();
const registryIds = registryModels.map((entry) => entry.modelId).sort();
const storageIds = listQuickStartModels().map((entry) => entry.modelId).sort();

assert.deepEqual(storageIds, registryIds);

for (const modelId of storageIds) {
  assert.ok(MODEL_REQUIREMENTS[modelId], `${modelId}: quickstart downloader requires preflight requirements`);
}

console.log('quickstart-storage-registry-sync.test: ok');
