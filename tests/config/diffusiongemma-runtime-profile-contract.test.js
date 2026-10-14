import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KNOWN_MODELS } from '../../src/models/diffusiongemma.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

for (const { modelId, defaultRuntimeProfile } of KNOWN_MODELS) {
  if (!defaultRuntimeProfile) {
    continue;
  }
  const profile = await readJson(`src/config/runtime/${defaultRuntimeProfile}.json`);
  assert.equal(profile.id, defaultRuntimeProfile, `${modelId}: advertised runtime profile must exist under its id`);
  assert.equal(profile.model, modelId, `${modelId}: advertised runtime profile must be scoped to the model`);
  const routing = profile.runtime?.inference?.moe?.routing ?? {};
  if (routing.activeExpertSelection === 'topk-route') {
    assert.equal(
      Object.hasOwn(routing, 'maxTokensPerExpert'),
      false,
      `${modelId}: topk-route profile must not carry the readback slot budget.`
    );
  }
}

console.log('diffusiongemma-runtime-profile-contract.test: ok');
